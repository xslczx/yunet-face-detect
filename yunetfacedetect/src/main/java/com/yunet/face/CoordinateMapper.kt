package com.yunet.face

import kotlin.math.max
import kotlin.math.min

/**
 * 网络输入坐标 → 调用方图像坐标的映射器。
 *
 * 映射链路：
 *   net 坐标 --(letterbox/stretch 逆变换)--> 竖直位图坐标
 *         --(EXIF 旋转逆变换)--> 解码位图坐标 --(× finalScale)--> 调用方坐标系
 *
 * letterbox 模式下内容贴左上角（官方 Pad 为右下补边），因此逆变换为 (x - padX) / scale。
 * EXIF 旋转逆变换按 Android Matrix.postRotate（顺时针为正）推导。
 */
internal class CoordinateMapper(
    private val resizeMode: ResizeMode,
    /** letterbox 均匀缩放（net → 竖直） */
    private val scale: Float,
    /** stretch 各轴缩放（net → 竖直） */
    private val scaleX: Float,
    private val scaleY: Float,
    private val padX: Float,
    private val padY: Float,
    /** 竖直位图相对解码位图顺时针旋转的角度（0/90/180/270） */
    private val rotationDeg: Float,
    /** 竖直位图尺寸（旋转后） */
    private val uprightW: Int,
    private val uprightH: Int,
    /** 最终坐标系尺寸（用于钳制） */
    private val finalW: Int,
    private val finalH: Int,
    /** 解码位图坐标 → 最终坐标系的放大倍数（文件采样为 sample；位图下采样为 1/inputScale） */
    private val finalScale: Float,
    /** 解码位图 → 竖直位图 是否先经过水平镜像（EXIF orientation 2/4/5/7） */
    private val mirrored: Boolean = false,
) {
    /**
     * 映射单个点，返回 [x, y]（已钳制在最终坐标系内）。
     *
     * 映射是转正变换的逆变换：转正 = 先水平镜像、后顺时针旋转 rotationDeg，
     * 因此逆映射先做旋转逆变换，再在解码坐标系内做水平镜像（x → W_d − x）。
     * rotationDeg == 0 时跳过旋转逆变换（写作“停在竖直坐标”的映射器）。
     */
    fun mapPoint(x: Float, y: Float): FloatArray {
        val ux = if (resizeMode == ResizeMode.LETTERBOX) (x - padX) / scale else x / scaleX
        val uy = if (resizeMode == ResizeMode.LETTERBOX) (y - padY) / scale else y / scaleY
        val deg = normalizeDeg(rotationDeg)
        var bx = ux
        var by = uy
        when (deg) {
            90f -> { bx = uy; by = uprightW - ux }
            180f -> { bx = uprightW - ux; by = uprightH - uy }
            270f -> { bx = uprightH - uy; by = ux }
        }
        if (mirrored) {
            // 旋转逆变换后的坐标系即“解码位图”坐标系；其宽度由旋转角度决定
            val decodedW = if (deg == 90f || deg == 270f) uprightH else uprightW
            bx = decodedW - bx
        }
        return floatArrayOf(
            (bx * finalScale).coerceIn(0f, finalW.toFloat()),
            (by * finalScale).coerceIn(0f, finalH.toFloat()),
        )
    }

    /** 映射矩形（四角分别映射后取外包矩形），返回 [left, top, right, bottom] */
    fun mapBox(left: Float, top: Float, right: Float, bottom: Float): FloatArray {
        val p0 = mapPoint(left, top)
        val p1 = mapPoint(right, top)
        val p2 = mapPoint(right, bottom)
        val p3 = mapPoint(left, bottom)
        val l = min(min(p0[0], p1[0]), min(p2[0], p3[0]))
        val t = min(min(p0[1], p1[1]), min(p2[1], p3[1]))
        val r = max(max(p0[0], p1[0]), max(p2[0], p3[0]))
        val b = max(max(p0[1], p1[1]), max(p2[1], p3[1]))
        return floatArrayOf(l, t, r, b)
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
