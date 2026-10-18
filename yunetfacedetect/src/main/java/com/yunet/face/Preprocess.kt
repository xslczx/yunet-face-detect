package com.yunet.face

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * 预处理器：调用方位图 -> 网络输入像素缓冲。
 *
 * 流水线（与官方评测 transforms 对齐）：
 * 1. HARDWARE 位图转 ARGB_8888（硬件位图无法读取像素）；
 * 2. 超大图下采样（maxInputImageDim，防 OOM），记录 inputScale 供坐标回溯；
 * 3. EXIF 转正：先水平镜像（orientation 2/4/5/7），再顺时针旋转 rotationDeg——
 *    顺序遵循 androidx ExifInterface 约定（镜像与旋转不可交换）；
 * 4. letterbox（等比 + 右下补 padValue，官方 Pad 语义）或 stretch 到网络输入尺寸；
 * 5. 按模型布局（NCHW/NHWC）与通道序（BGR/RGB）提取像素，
 *    归一化 out = pixel * normalizeScale + normalizeMean（官方为 1/0，即原始 0-255）。
 *
 * 注意：仅回收本类创建的中间位图，绝不回收调用方传入的位图。
 */
internal class Preprocessor(
    private val config: YuNetConfig,
    private val channels: Int,
    private val layout: InputLayout,
    private val netW: Int,
    private val netH: Int,
) {

    class Result(
        /** 按布局与通道序排列的像素值，长度 = channels * netW * netH */
        val pixels: FloatArray,
        /** letterbox 均匀缩放（net -> 竖直） */
        val scale: Float,
        /** stretch 各轴缩放（net -> 竖直） */
        val scaleX: Float,
        val scaleY: Float,
        val padX: Float,
        val padY: Float,
        /** 传入位图 -> 竖直位图 的整体缩小倍数（>=1，1 表示未缩小） */
        val inputScale: Float,
        /** 竖直位图相对传入位图顺时针旋转的角度（0/90/180/270） */
        val rotationDeg: Float,
        /** 竖直位图是否相对传入位图水平镜像（EXIF orientation 2/4/5/7） */
        val mirrored: Boolean,
        val uprightW: Int,
        val uprightH: Int,
        val sourceW: Int,
        val sourceH: Int,
    )

    fun process(source: Bitmap, exifDegrees: Float): Result =
        process(source, ExifTransform.of(exifDegrees.roundToInt(), false))

    fun process(source: Bitmap, exif: ExifTransform): Result {
        require(!source.isRecycled) { "源位图已回收" }
        var current: Bitmap = source
        var owned = false

        // 1) HARDWARE 位图（如相机 SurfaceTexture 产出）无法 getPixels，先转换
        if (current.config == Bitmap.Config.HARDWARE) {
            val copied = current.copy(Bitmap.Config.ARGB_8888, false)
                ?: throw YuNetInputException("HARDWARE 位图转换 ARGB_8888 失败")
            current = copied
            owned = true
        }

        // 2) 超大图下采样
        var inputScale = 1f
        if (config.maxInputImageDim > 0) {
            val maxDim = max(current.width, current.height)
            if (maxDim > config.maxInputImageDim) {
                val f = config.maxInputImageDim.toFloat() / maxDim
                val nw = max(1, (current.width * f).roundToInt())
                val nh = max(1, (current.height * f).roundToInt())
                val scaled = Bitmap.createScaledBitmap(current, nw, nh, true)
                if (owned) current.recycle()
                current = scaled
                owned = true
                inputScale = maxDim.toFloat() / max(nw, nh)
            }
        }

        // 3) EXIF 转正：先水平镜像，再旋转（androidx ExifInterface 约定顺序，不可交换）
        val rotationDeg = normalizeDeg(exif.rotationDegrees.toFloat())
        if (exif.mirrored) {
            val matrix = Matrix()
            matrix.postScale(-1f, 1f)
            val flipped = Bitmap.createBitmap(
                current, 0, 0, current.width, current.height, matrix, true,
            )
            if (owned) current.recycle()
            current = flipped
            owned = true
        }
        if (rotationDeg != 0f) {
            val matrix = Matrix().apply { postRotate(rotationDeg) }
            val rotated = Bitmap.createBitmap(current, 0, 0, current.width, current.height, matrix, true)
            if (owned) current.recycle()
            current = rotated
            owned = true
        }

        val uprightW = current.width
        val uprightH = current.height

        // 4) 缩放到网络输入
        var scale = 1f
        var scaleX = 1f
        var scaleY = 1f
        val netBitmap: Bitmap = when (config.resizeMode) {
            ResizeMode.LETTERBOX -> {
                scale = min(netW.toFloat() / uprightW, netH.toFloat() / uprightH)
                val nw = max(1, (uprightW * scale).roundToInt())
                val nh = max(1, (uprightH * scale).roundToInt())
                val resized = Bitmap.createScaledBitmap(current, nw, nh, true)
                val target = Bitmap.createBitmap(netW, netH, Bitmap.Config.ARGB_8888)
                val canvas = Canvas(target)
                canvas.drawColor(padArgb())
                canvas.drawBitmap(resized, 0f, 0f, null)
                if (resized !== current) resized.recycle()
                target
            }
            ResizeMode.STRETCH -> {
                scaleX = netW.toFloat() / uprightW
                scaleY = netH.toFloat() / uprightH
                Bitmap.createScaledBitmap(current, netW, netH, true)
            }
        }

        // 5) 提取像素（必须在任何 recycle 之前）
        val pixels = extractPixels(netBitmap)
        if (netBitmap !== current) netBitmap.recycle()
        if (owned) current.recycle()

        return Result(
            pixels = pixels,
            scale = scale,
            scaleX = scaleX,
            scaleY = scaleY,
            padX = 0f,
            padY = 0f,
            inputScale = inputScale,
            rotationDeg = rotationDeg,
            mirrored = exif.mirrored,
            uprightW = uprightW,
            uprightH = uprightH,
            sourceW = source.width,
            sourceH = source.height,
        )
    }

    private fun extractPixels(bitmap: Bitmap): FloatArray {
        val w = bitmap.width
        val h = bitmap.height
        val n = w * h
        val px = IntArray(n)
        bitmap.getPixels(px, 0, w, 0, 0, w, h)
        val out = FloatArray(channels * n)
        val scale = config.normalizeScale
        val mean = config.normalizeMean
        val bgr = config.channelOrder == ChannelOrder.BGR
        if (channels == 3) {
            if (layout == InputLayout.NCHW) {
                for (i in 0 until n) {
                    val p = px[i]
                    val r = ((p shr 16) and 0xFF).toFloat()
                    val g = ((p shr 8) and 0xFF).toFloat()
                    val b = (p and 0xFF).toFloat()
                    out[i] = (if (bgr) b else r) * scale + mean
                    out[n + i] = g * scale + mean
                    out[2 * n + i] = (if (bgr) r else b) * scale + mean
                }
            } else {
                for (i in 0 until n) {
                    val p = px[i]
                    val r = ((p shr 16) and 0xFF).toFloat()
                    val g = ((p shr 8) and 0xFF).toFloat()
                    val b = (p and 0xFF).toFloat()
                    out[i * 3] = (if (bgr) b else r) * scale + mean
                    out[i * 3 + 1] = g * scale + mean
                    out[i * 3 + 2] = (if (bgr) r else b) * scale + mean
                }
            }
        } else {
            // 灰度：ITU-R 601 luma
            for (i in 0 until n) {
                val p = px[i]
                val r = (p shr 16) and 0xFF
                val g = (p shr 8) and 0xFF
                val b = p and 0xFF
                out[i] = ((77 * r + 150 * g + 29 * b) shr 8).toFloat() * scale + mean
            }
        }
        return out
    }

    private fun padArgb(): Int {
        val pv = config.padValue
        return (0xFF shl 24) or (pv shl 16) or (pv shl 8) or pv
    }

    private fun normalizeDeg(deg: Float): Float {
        val d = ((deg % 360f) + 360f) % 360f
        return when {
            d < 45f || d >= 315f -> 0f
            d < 135f -> 90f
            d < 225f -> 180f
            else -> 270f
        }
    }
}
