package com.yunet.face.demo

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.BitmapShader
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.RectF
import android.graphics.Shader
import android.util.Log
import androidx.exifinterface.media.ExifInterface
import com.yunet.face.Face
import com.yunet.face.YuNetConfig
import com.yunet.face.YuNetDetector
import java.io.File
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext

/**
 * 人脸裁剪：把用户选的照片裁成以最大人脸为中心的 3:4 半身照。
 *
 * 1. 取最大人脸，按双眼连线角度做水平校正（死区 3° 内不转、上限 25°，
 *    死区挡关键点噪声，上限防误识别的离谱角度）；
 * 2. 旋转子图后重新检测，拿旋转坐标系下的框（避免手算旋转框误差）；
 * 3. 人脸框外扩：左右 ±0.70 脸宽、发顶 +0.90 脸高、肩线 +1.60 脸高，再调 3:4；
 * 4. 钳制到图像边界。
 *
 * 旋转用 BitmapShader(CLAMP) 平铺边缘像素填充越界区（等价 OpenCV BORDER_REPLICATE，无黑角）。
 * 任何一步失败返回 null，由调用方回退原图，绝不阻塞建档流程。
 */
object ArchiveFaceCrop {

    private const val SCORE_THRESHOLD = 0.3f

    /**
     * 构图参数。各留白均为「相对人脸检测框」的倍数。
     *
     * 注意人脸框约是眉毛到下巴的范围，头发在框外，头顶留白按脸框高的倍数给。
     * 调大留白 = 人物更小、空间更多；调小 = 人物更满。
     */
    data class Params(
        /** 输出宽高比 w/h，默认 3:4 */
        val aspect: Float = 0.75f,
        /** 头顶留白：脸框上沿以上保留 ×脸框高（想要更多头发上方空间就调大） */
        val headroomScale: Float = 0.90f,
        /** 下方留白：脸框下沿以下保留 ×脸框高（胸口/半身范围） */
        val chinroomScale: Float = 1.60f,
        /** 左右留白：×脸框宽 */
        val sideScale: Float = 0.70f,
        /** 倾斜死区（度）：小于该角度不旋转（模型关键点噪声约 ±3°） */
        val tiltDeadZoneDeg: Float = 3f,
        /** 倾斜校正上限（度）：防误识别关键点导致过转 */
        val tiltMaxDeg: Float = 25f,
    ) {
        init {
            require(aspect in 0.5f..1.1f) { "aspect 应在 0.5~1.1" }
            require(headroomScale in 0f..2.5f) { "headroomScale 应在 0~2.5" }
            require(chinroomScale in 0f..3.5f) { "chinroomScale 应在 0~3.5" }
            require(sideScale in 0f..2.5f) { "sideScale 应在 0~2.5" }
            require(tiltDeadZoneDeg in 0f..10f) { "tiltDeadZoneDeg 应在 0~10" }
            require(tiltMaxDeg in tiltDeadZoneDeg..45f) { "tiltMaxDeg 应在死区~45" }
        }
    }

    /**
     * 旋转后重检的脸面积下限（相对原脸面积比例）。旋转不改变脸的尺寸，
     * 低于该比例的框视为 CLAMP 平铺边缘 / 背景纹理产生的误检，不参与选脸。
     */
    private const val MIN_REDETECT_AREA_RATIO = 0.25f

    private const val JPEG_QUALITY = 92

    /** 纯色边条判定：整行/列像素 RGB 最大通道低于此值视为边条（JPEG 压缩下黑边典型 ≤5） */
    private const val BAR_CHANNEL_MAX = 30

    /** 每侧最多裁除的比例（防止把本就以深色开头的真实照片内容整块吃掉） */
    private const val MAX_TRIM_RATIO = 0.2f

    /** 裁边后内容的最小尺寸，小于此值放弃裁边（视为异常图，保持原样） */
    private const val MIN_CONTENT_PX = 256

    /** 旋转后越出原照片区域的中性底色（浅灰，档案照影棚背景风格） */
    private const val FILL_COLOR = 0xFFF2F2F2.toInt()

    /** 与库默认模型一致（app/src/main/assets，MainActivity.MODEL_ASSET 同名） */
    private const val MODEL_ASSET = "yunet_fp16.tflite"

    @Volatile
    private var detector: YuNetDetector? = null
    private val mutex = Mutex()

    /**
     * 裁剪 [imagePath] 指向的照片，结果写到与其同目录的 crop_<name> 文件。
     *
     * @param report 过程进度 / 失败原因回调（IO 线程调用，用于诊断展示），默认忽略
     * @return 裁剪后文件路径；未检测到人脸 / 检测或解码失败时返回 null（调用方回退原图）。
     */
    suspend fun crop(
        context: Context,
        imagePath: String,
        params: Params = Params(),
        report: (String) -> Unit = {},
    ): String? = withContext(Dispatchers.IO) {
        runCatching {
            val d = ensureDetector(context)
            if (d == null) {
                report("检测器创建失败（模型加载异常）")
                return@runCatching null
            }
            val bitmap0 = decodeSampled(imagePath, maxDim = 2048)
            if (bitmap0 == null) {
                report("图片解码失败: $imagePath")
                return@runCatching null
            }
            // 部分来源图片自带左右纯色边条（如 AI 工具导出的 1080 内容 + 黑边合成）：
            // 不先裁掉，旋转校正后边条会斜切进出图，看起来像「图歪了 + 多了黑色内容」。
            val bitmap = trimUniformEdgeBars(bitmap0, report).also { if (it !== bitmap0) bitmap0.recycle() }
            val cropped = bitmap.use {
                val faces = d.detect(it)
                val face = faces.maxByOrNull { f -> f.width * f.height }
                if (face == null) {
                    report("未检出人脸（阈值 $SCORE_THRESHOLD）")
                    null
                } else {
                    cropInternal(d, it, face, params, report)
                }
            }
            if (cropped == null) return@runCatching null
            val saved = cropped.saveJpeg(cropFileFor(imagePath))?.absolutePath
            if (saved == null) report("裁剪结果保存失败")
            saved
        }.getOrNull()
    }

    /** 与源照片同目录的裁剪输出文件，随 archive_photos 目录一并清理。 */
    private fun cropFileFor(imagePath: String): File {
        val source = File(imagePath)
        return File(source.parentFile, "crop_${source.name}")
    }

    /**
     * 裁除照片四周的均匀纯色边条（pillarbox / letterbox 常见于工具导出图）。
     *
     * 判定：整行或整列像素 RGB 最大通道均低于 [BAR_CHANNEL_MAX] 才算边条，
     * 每侧最多裁 [MAX_TRIM_RATIO]；裁后内容小于 [MIN_CONTENT_PX] 或无可裁则原样返回。
     * 无边条时零拷贝返回原图（createBitmap 全尺寸会返回原引用）。
     */
    private fun trimUniformEdgeBars(src: Bitmap, report: (String) -> Unit): Bitmap {
        val w = src.width
        val h = src.height

        fun barPixel(p: Int): Boolean =
            maxOf((p shr 16) and 0xFF, (p shr 8) and 0xFF, p and 0xFF) < BAR_CHANNEL_MAX

        fun colIsBar(x: Int): Boolean {
            val col = IntArray(h)
            src.getPixels(col, 0, 1, x, 0, 1, h)
            return col.all { barPixel(it) }
        }

        fun rowIsBar(y: Int): Boolean {
            val row = IntArray(w)
            src.getPixels(row, 0, w, 0, y, w, 1)
            return row.all { barPixel(it) }
        }

        val maxH = (w * MAX_TRIM_RATIO).toInt()
        val maxV = (h * MAX_TRIM_RATIO).toInt()
        var left = 0
        while (left < maxH && colIsBar(left)) left++
        var right = w - 1
        while (right > w - 1 - maxH && colIsBar(right)) right--
        var top = 0
        while (top < maxV && rowIsBar(top)) top++
        var bottom = h - 1
        while (bottom > h - 1 - maxV && rowIsBar(bottom)) bottom--

        val cw = right - left + 1
        val ch = bottom - top + 1
        if (left == 0 && top == 0 && cw == w && ch == h) return src
        if (cw < MIN_CONTENT_PX || ch < MIN_CONTENT_PX) return src
        val msg = "已识别并裁除图片自带黑边: 左${left} 右${w - 1 - right} 上${top} 下${h - 1 - bottom}px，内容 ${cw}x${ch}"
        Log.i("YuNetCrop", "TRIM bars l=$left r=${w - 1 - right} t=$top b=${h - 1 - bottom} content=${cw}x${ch}")
        report(msg)
        return Bitmap.createBitmap(src, left, top, cw, ch)
    }

    private suspend fun cropInternal(
        detector: YuNetDetector,
        bitmap: Bitmap,
        face: Face,
        params: Params,
        report: (String) -> Unit = {},
    ): Bitmap? {
        // 倾斜角保守化：死区 + 上限（eyeAngleDegrees 不依赖关键点命名顺序）
        val tiltDeg = face.eyeAngleDegrees()
        val rotateDeg = when {
            abs(tiltDeg) < params.tiltDeadZoneDeg -> 0f
            else -> tiltDeg.coerceIn(-params.tiltMaxDeg, params.tiltMaxDeg)
        }

        // 1) 在原图坐标算带旋转余量的子图：先按外扩参数算 3:4 裁剪框
        val base = paddedAspectRect(face.box, bitmap.width, bitmap.height, params)
        // [TEMP-DEBUG]
        Log.i("YuNetCrop", "STEP1 bitmap=${bitmap.width}x${bitmap.height} box=[${face.box.left},${face.box.top},${face.box.right},${face.box.bottom}] tilt=$tiltDeg rotate=$rotateDeg base=[${base.left},${base.top},${base.right},${base.bottom}]")
        // 旋转余量：对角线 × sin(半角)。kotlin.math.sin 收弧度——度数必须先 toRadians，
        // 否则小倾角会把子图撑成整图、大倾角(≥7°)更会算出负 margin 把子图收缩成人脸细条
        val margin = (hypot(base.width.toDouble(), base.height.toDouble()) *
            sin(Math.toRadians(abs(rotateDeg).toDouble()) / 2.0) + 4f).toFloat()
        val subRect = base.expand(margin).intersect(0f, 0f, bitmap.width.toFloat(), bitmap.height.toFloat())
        if (subRect.width < 8 || subRect.height < 8) return null
        val sub = Bitmap.createBitmap(
            bitmap,
            subRect.left.toInt(),
            subRect.top.toInt(),
            subRect.width.toInt(),
            subRect.height.toInt(),
        )

        // 2) 旋转子图（CLAMP 平铺边缘，无黑角），转回的角度与 Android 坐标系相反
        val rotated = if (rotateDeg != 0f) {
            rotateWithNeutralFill(sub, -rotateDeg).also { if (it !== sub) sub.recycle() }
        } else {
            sub
        }
        // [TEMP-DEBUG]
        Log.i("YuNetCrop", "STEP2 sub=${subRect.width.toInt()}x${subRect.height.toInt()} rotated=${rotated.width}x${rotated.height}")

        // 3) 旋转后重新检测，在旋转坐标系里再裁一次；
        //    重检偶发丢失（fp16 关键点噪声 / 边缘小脸）时回退子图中央 3:4，不让首次已检出的人脸作废
        return try {
            val rotatedFaces = detector.detect(rotated)
            // 只从「面积 ≥ 原脸 × 比例」的候选里选最大脸：旋转不会缩小人脸，
            // 尺寸过小的框是 CLAMP 平铺边缘/背景纹理的误检，不能替代真实脸
            val origArea = face.box.width() * face.box.height()
            val rf = rotatedFaces
                .filter { f -> f.width * f.height >= origArea * MIN_REDETECT_AREA_RATIO }
                .maxByOrNull { f -> f.width * f.height }
            // [TEMP-DEBUG]
            Log.i("YuNetCrop", "STEP3 rotatedFaces=${rotatedFaces.size} " +
                "origArea=$origArea candidates=" +
                rotatedFaces.filter { f -> f.width * f.height >= origArea * MIN_REDETECT_AREA_RATIO }.size +
                " rfBox=${rf?.box?.left},${rf?.box?.top},${rf?.box?.right},${rf?.box?.bottom}")
            val target0 = if (rf != null) {
                paddedAspectRect(rf.box, rotated.width, rotated.height, params)
            } else {
                // 子图 = 原帧 3:4 基准框外扩 margin 所得、与基准框同心，中央 3:4 仍以人脸为中心
                report("旋转后重检未检出，回退子图中央 3:4 区域")
                centerAspectRect(rotated.width, rotated.height, params.aspect)
            }

            // 4) 取景框收敛：完全落在照片覆盖范围内（出图无补底/黑边），
            //    保留头顶发际空间（宁可收紧下方胸口，不裁半张头发），人脸保持水平 3:4。
            //    极端情况收敛失败时退回居中内接安全区。
            val safe = photoSafeRect(subRect.width.toInt(), subRect.height.toInt(), rotateDeg, rotated.width, rotated.height, params.aspect)
            val fallback = run {
                var l = max(target0.left, safe.left)
                var t = max(target0.top, safe.top)
                var r = min(target0.right, safe.right)
                var b = min(target0.bottom, safe.bottom)
                var w2 = r - l
                var h2 = b - t
                if (w2 / h2 > params.aspect) w2 = h2 * params.aspect else h2 = w2 / params.aspect
                val fcx = rf?.centerX ?: rotated.width / 2f
                val fcy = rf?.centerY ?: rotated.height / 2f
                l = (fcx - w2 / 2f).coerceIn(l, r - w2)
                t = (fcy - h2 / 2f).coerceIn(t, b - h2)
                CropRect(l, t, l + w2, t + h2)
            }
            val target = if (rf != null) {
                fitFrameInPhoto(
                    desired = faceAspectRect(
                        rf.box,
                        params.sideScale,
                        params.headroomScale,
                        params.chinroomScale,
                        params.aspect,
                    ),
                    rfBox = rf.box,
                    subW = subRect.width.toInt(),
                    subH = subRect.height.toInt(),
                    rotateDeg = rotateDeg,
                    canvasW = rotated.width,
                    canvasH = rotated.height,
                    aspect = params.aspect,
                    fallback = fallback,
                )
            } else {
                fallback
            }
            // [TEMP-DEBUG]
            Log.i("YuNetCrop", "STEP4 final=[${target.left},${target.top},${target.right},${target.bottom}]" +
                " fallback=[${fallback.left},${fallback.top},${fallback.right},${fallback.bottom}]")
            Bitmap.createBitmap(
                rotated,
                target.left.toInt().coerceAtLeast(0),
                target.top.toInt().coerceAtLeast(0),
                target.width.toInt().coerceAtMost(rotated.width - target.left.toInt().coerceAtLeast(0)),
                target.height.toInt().coerceAtMost(rotated.height - target.top.toInt().coerceAtLeast(0)),
            )
        } finally {
            if (rotated !== sub) rotated.recycle()
        }
    }

    /**
     * 在「旋转后照片」覆盖范围内收敛 [aspect] 比例取景框。
     *
     * 1. 初值 = 期望外扩框（按 [Params] 留白）钳进画布；
     * 2. 保持人脸在框内的相对位置，按 0.97 步长收缩，直到四角逆旋转后都落回子图矩形
     *    （= 取景框完全在照片内，出图不含任何补底）；
     * 3. 在照片允许范围内尽量上移取景框保住头顶发际——上移同时收紧下方区域；
     * 4. 收缩到人脸都放不下、或不收敛时返回 [fallback]（居中内接安全区）。
     */
    private fun fitFrameInPhoto(
        desired: CropRect,
        rfBox: RectF,
        subW: Int,
        subH: Int,
        rotateDeg: Float,
        canvasW: Int,
        canvasH: Int,
        aspect: Float,
        fallback: CropRect,
    ): CropRect {
        val fcx = rfBox.centerX()
        val fcy = rfBox.centerY()
        val relX = ((fcx - desired.left) / desired.width).coerceIn(0.15f, 0.85f)
        val relY = ((fcy - desired.top) / desired.height).coerceIn(0.05f, 0.85f)
        var h = min(desired.height, canvasH.toFloat())
        var w = h * aspect
        var iter = 0
        fun place(height: Float): CropRect {
            val width = height * aspect
            return CropRect(fcx - relX * width, fcy - relY * height, fcx + (1 - relX) * width, fcy + (1 - relY) * height)
        }
        fun fits(f: CropRect) = cornersInPhoto(f.left, f.top, f.width, f.height, subW, subH, rotateDeg, canvasW, canvasH)
        while (iter++ < 96 && !fits(place(h))) h *= 0.97f
        val frame = place(h)
        if (!fits(frame) || rfBox.width() + 8f > frame.width || rfBox.height() + 8f > frame.height) {
            // 收缩到极限仍容不下人脸，退回安全区
            return fallback
        }
        // 上移保头顶：每次 1px，直到顶角触照片边界，且人脸仍完整在框内
        var guard = 0
        var top = frame.top
        while (guard++ < frame.height.toInt() &&
            top + frame.height - 1f >= rfBox.bottom + 4f &&
            cornersInPhoto(frame.left, top - 1f, frame.width, frame.height, subW, subH, rotateDeg, canvasW, canvasH)
        ) {
            top -= 1f
        }
        return CropRect(frame.left, top, frame.left + frame.width, top + frame.height)
    }

    /**
     * 轴对齐取景框四角是否都落在「旋转后照片」覆盖范围内。
     * 画布点按 rotateWithNeutralFill 的逆变换映射回子图坐标后判断是否在子图矩形内。
     */
    private fun cornersInPhoto(
        left: Float,
        top: Float,
        width: Float,
        height: Float,
        subW: Int,
        subH: Int,
        rotateDeg: Float,
        canvasW: Int,
        canvasH: Int,
    ): Boolean {
        val theta = Math.toRadians(rotateDeg.toDouble())
        val c = cos(theta)
        val s = sin(theta)
        for (px in floatArrayOf(left, left + width)) {
            for (py in floatArrayOf(top, top + height)) {
                val u = px - canvasW / 2f
                val v = py - canvasH / 2f
                val qx = (c * u - s * v) + subW / 2f
                val qy = (s * u + c * v) + subH / 2f
                if (qx < -0.5f || qx > subW + 0.5f || qy < -0.5f || qy > subH + 0.5f) return false
            }
        }
        return true
    }

    /** 人脸外扩期望框（不钳制边界）：左右/上/下留白均为×脸框倍数，再调成 [aspect] 比例。 */
    private fun faceAspectRect(
        box: RectF,
        sideScale: Float,
        headroomScale: Float,
        chinroomScale: Float,
        aspect: Float,
    ): CropRect {
        var left = box.left - sideScale * box.width()
        var right = box.right + sideScale * box.width()
        var top = box.top - headroomScale * box.height()
        var bottom = box.bottom + chinroomScale * box.height()

        val cx = box.left + box.width() / 2f
        val cy = box.top + box.height() / 2f
        var height = bottom - top
        var width = height * aspect
        if (width < right - left) {
            width = right - left
            height = width / aspect
        }
        left = cx - width / 2f
        right = cx + width / 2f
        top = cy - (cy - top) / (bottom - top) * height
        bottom = top + height
        return CropRect(left, top, right, bottom)
    }

    /** 位图中央的 [aspect] 比例区域（尽量取满短边），用于旋转后重检失败的回退。 */
    private fun centerAspectRect(w: Int, h: Int, aspect: Float): CropRect {
        var cw = w.toFloat()
        var ch = cw / aspect
        if (ch > h) {
            ch = h.toFloat()
            cw = ch * aspect
        }
        val left = (w - cw) / 2f
        val top = (h - ch) / 2f
        return CropRect(left, top, left + cw, top + ch)
    }

    /**
     * 旋转画布中「照片必然覆盖」的居中内接矩形（画布坐标）。
     *
     * 子图 [subW]x[subH] 被旋转 [rotateDeg] 度后居中放在画布 [canvasW]x[canvasH] 里，
     * 越出照片的四个楔形是底色。要求内接矩形（3:4）的角点旋转回原照片后仍落在照片内：
     * 按 a = 0.75b 求解 b 的上限，得到尽量大且必为照片内容的取景区。
     */
    private fun photoSafeRect(
        subW: Int,
        subH: Int,
        rotateDeg: Float,
        canvasW: Int,
        canvasH: Int,
        aspect: Float,
    ): CropRect {
        val theta = Math.toRadians(abs(rotateDeg).toDouble())
        val c = cos(theta)
        val s = sin(theta)
        // 半宽/半高约束：a·cosθ + b·sinθ ≤ subW/2 且 b·cosθ − a·sinθ ≤ subH/2，其中 a = aspect·b
        val b = min(
            (subW / 2.0) / (aspect * c + s),
            (subH / 2.0) / (c - aspect * s),
        ).toFloat()
        val a = aspect * b
        val cx = canvasW / 2f
        val cy = canvasH / 2f
        return CropRect(cx - a, cy - b, cx + a, cy + b)
    }

    /** 人脸框外扩并调成 [Params.aspect] 比例，钳制在图像边界内（w/h 均已保证 >= 1）。 */
    private fun paddedAspectRect(
        box: RectF,
        imageWidth: Int,
        imageHeight: Int,
        params: Params,
    ): CropRect {
        val d = faceAspectRect(box, params.sideScale, params.headroomScale, params.chinroomScale, params.aspect)
        var left = d.left
        var right = d.right
        var top = d.top
        var bottom = d.bottom

        // 钳制 + 越界侧反向补偿，尽量保住 3:4
        val w = imageWidth.toFloat()
        val h = imageHeight.toFloat()
        if (left < 0f) {
            right = min(w, right - left)
            left = 0f
        }
        if (right > w) {
            left = max(0f, left - (right - w))
            right = w
        }
        if (top < 0f) {
            bottom = min(h, bottom - top)
            top = 0f
        }
        if (bottom > h) {
            top = max(0f, top - (bottom - h))
            bottom = h
        }
        return CropRect(left, top, right, bottom)
    }

    private class CropRect(val left: Float, val top: Float, val right: Float, val bottom: Float) {
        val width: Float get() = right - left
        val height: Float get() = bottom - top

        fun expand(margin: Float) = CropRect(left - margin, top - margin, right + margin, bottom + margin)

        fun intersect(l: Float, t: Float, r: Float, b: Float) = CropRect(
            max(left, l), max(top, t), min(right, r), min(bottom, b),
        )
    }

    /**
     * 旋转位图并返回包含全部内容的扩展画布。
     * 越出原照片的楔形区用中性浅灰底填充（档案照风格，类似影棚背景）——
     * 之前用 BitmapShader(CLAMP) 平铺边缘像素，出图角落会出现拖影条纹，观感像「歪」。
     */
    private fun rotateWithNeutralFill(source: Bitmap, degrees: Float): Bitmap {
        val radians = Math.toRadians(abs(degrees).toDouble())
        val outW = (source.width * cos(radians) + source.height * sin(radians)).toInt()
            .coerceAtLeast(source.width)
        val outH = (source.width * sin(radians) + source.height * cos(radians)).toInt()
            .coerceAtLeast(source.height)
        val out = Bitmap.createBitmap(outW, outH, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(out)
        canvas.drawColor(FILL_COLOR)
        val paint = Paint(Paint.FILTER_BITMAP_FLAG)
        val shader = BitmapShader(source, Shader.TileMode.CLAMP, Shader.TileMode.CLAMP)
        paint.shader = shader
        canvas.translate((outW - source.width) / 2f, (outH - source.height) / 2f)
        canvas.rotate(degrees, source.width / 2f, source.height / 2f)
        // 只画位图本身占据的区域，越界楔形交给上面的中性底色（CLAMP 仅用于消除边缘缝隙）
        canvas.drawRect(0f, 0f, source.width.toFloat(), source.height.toFloat(), paint)
        return out
    }

    private fun Bitmap.saveJpeg(target: File): File? = runCatching {
        target.parentFile?.mkdirs()
        target.outputStream().use { out ->
            compress(Bitmap.CompressFormat.JPEG, JPEG_QUALITY, out)
        }
        target
    }.also { recycle() }.getOrNull()

    private suspend fun ensureDetector(context: Context): YuNetDetector? {
        detector?.let { return it }
        return runCatching {
            mutex.withLock {
                detector ?: YuNetDetector.create(context.applicationContext, buildConfig())
                    .also { detector = it }
            }
        }.getOrNull()
    }

    /** 阈值在配置期固定（新 API 不再支持逐次检测传阈值）；取 README §5 推荐 0.3 下限，宁可多检由「选最大脸」兜底，避免真实照片漏检。 */
    private fun buildConfig(): YuNetConfig = YuNetConfig.Builder()
        .useAsset(MODEL_ASSET)
        .scoreThreshold(SCORE_THRESHOLD)
        .topK(10) // 保留若干候选，仍按面积选最大脸
        .maxInputImageDim(2048)
        .build()

    private fun decodeSampled(path: String, maxDim: Int): Bitmap? = runCatching {
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeFile(path, bounds)
        if (bounds.outWidth <= 0 || bounds.outHeight <= 0) return null
        var sample = 1
        while (maxOf(bounds.outWidth, bounds.outHeight) / (sample * 2) >= maxDim) sample *= 2
        BitmapFactory.decodeFile(path, BitmapFactory.Options().apply { inSampleSize = sample })
            ?.let { uprightByExif(it, path) }
    }.getOrNull()

    /** 按 EXIF 方向转正（与库内 detectFileSync 行为一致）；YuNet 只认近似正脸，侧躺/倒置图会 0 检出。 */
    private fun uprightByExif(bitmap: Bitmap, path: String): Bitmap {
        val deg = runCatching { ExifInterface(path).rotationDegrees.toFloat() }.getOrDefault(0f)
        if (deg == 0f) return bitmap
        val m = Matrix().apply { postRotate(deg) }
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, m, true)
            .also { if (it !== bitmap) bitmap.recycle() }
    }

    private inline fun <R> Bitmap.use(block: (Bitmap) -> R): R {
        try {
            return block(this)
        } finally {
            recycle()
        }
    }
}