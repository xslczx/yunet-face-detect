package com.yunet.face

import android.graphics.PointF
import android.graphics.RectF
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * 单张人脸检测结果。所有坐标均为**调用方传入图像**（detectFile 时为文件全分辨率）的像素坐标。
 *
 * landmark 布局（10 个 float，[x0,y0, x1,y1, x2,y2, x3,y3, x4,y4]），
 * 顺序依据训练标注 labelv2（SCRFD 约定）推断为：
 * 0=左眼、1=右眼、2=鼻尖、3=左嘴角、4=右嘴角（解剖学左右；正脸时 0 号点在图中偏右）。
 *
 * 注意：该命名顺序为约定推断，首次集成请用正脸照片配合 YuNetDiagnostics 目视验证。
 * 如需不依赖命名的水平校正，请使用 [eyeAngleDegrees]。
 */
class Face(
    /** 人脸框（x1,y1,x2,y2） */
    val box: RectF,
    /** 人脸置信度 = cls × obj，范围 [0,1] */
    val score: Float,
    /** 5 个关键点（10 个 float），未启用或解码失败时为 null */
    val landmarks: FloatArray?,
) {
    @JvmField
    val width: Float = box.width()
    @JvmField
    val height: Float = box.height()

    val centerX: Float get() = box.centerX()
    val centerY: Float get() = box.centerY()

    /** 取第 index 个关键点（0..4），越界或无关键点时返回 null */
    fun point(index: Int): PointF? {
        val l = landmarks ?: return null
        if (index < 0 || index >= LANDMARK_COUNT) return null
        return PointF(l[index * 2], l[index * 2 + 1])
    }

    val leftEye: PointF? get() = point(0)
    val rightEye: PointF? get() = point(1)
    val nose: PointF? get() = point(2)
    val mouthLeft: PointF? get() = point(3)
    val mouthRight: PointF? get() = point(4)

    /**
     * 双眼连线倾角（度）。以图中 x 较小（偏左）的眼为起点指向另一只眼，逆时针为正，范围 (-180, 180]。
     * 不依赖 landmark 命名顺序，可直接用于档案照水平校正。
     * 无关键点时返回 0。
     */
    fun eyeAngleDegrees(): Float {
        val l = landmarks ?: return 0f
        val ax = l[0]; val ay = l[1]
        val bx = l[2]; val by = l[3]
        val (sx, sy) = if (ax <= bx) ax to ay else bx to by
        val (ex, ey) = if (ax <= bx) bx to by else ax to ay
        return Math.toDegrees(atan2((ey - sy).toDouble(), (ex - sx).toDouble())).toFloat()
    }

    /**
     * 5 个关键点是否通过几何一致性校验（纯几何，可在 JVM 单测覆盖）。
     *
     * 校验（宽松阈值，只为拦截“关键点错乱/乱码”，不拦截正常侧脸）：
     * 1. 双眼、双嘴角间距非零；
     * 2. 嘴部连线长度未严重偏离双眼连线（防关键点角色互换）；
     * 3. 鼻尖近似落在双眼连线中点与嘴部连线中点的连线上（垂距 ≤ 0.85×线段长），
     *    且在该线段的合理延长范围内（投影比 -0.5~1.5）。
     *
     * 该标志为提示性判据：为 false 时建议消费方（如校平、裁剪）回退到
     * 不依赖关键点的策略。坐标值本身不会被修改。
     */
    fun isLandmarksConsistent(): Boolean = landmarksConsistent(landmarks)

    override fun toString(): String =
        "Face(box=[${box.left},${box.top},${box.right},${box.bottom}], score=$score, " +
            "eyeAngle=${eyeAngleDegrees()}deg)"

    companion object {
        const val LANDMARK_COUNT = 5

        /** 关键点几何一致性校验的纯函数实现（不依赖 Android 类，便于 JVM 单测） */
        @JvmStatic
        fun landmarksConsistent(landmarks: FloatArray?): Boolean {
            if (landmarks == null || landmarks.size < 10) return false
            val ex1 = landmarks[0]; val ey1 = landmarks[1]
            val ex2 = landmarks[2]; val ey2 = landmarks[3]
            val nx = landmarks[4]; val ny = landmarks[5]
            val mx1 = landmarks[6]; val my1 = landmarks[7]
            val mx2 = landmarks[8]; val my2 = landmarks[9]

            val eyeDist = hypot(ex1 - ex2, ey1 - ey2)
            val mouthDist = hypot(mx1 - mx2, my1 - my2)
            if (eyeDist < 1e-4f || mouthDist < 1e-4f) return false
            if (mouthDist < 0.15f * eyeDist || mouthDist > 5f * eyeDist) return false

            val eyeMidX = (ex1 + ex2) / 2f; val eyeMidY = (ey1 + ey2) / 2f
            val mouthMidX = (mx1 + mx2) / 2f; val mouthMidY = (my1 + my2) / 2f
            val segX = mouthMidX - eyeMidX; val segY = mouthMidY - eyeMidY
            val segLen2 = segX * segX + segY * segY
            if (segLen2 < 1e-8f) return false

            // 鼻尖到“双眼中点-嘴中点”连线的垂距（点到直线距离）
            val cross = segX * (ny - eyeMidY) - segY * (nx - eyeMidX)
            val perpDist = abs(cross) / sqrt(segLen2)
            if (perpDist > 0.85f * sqrt(segLen2)) return false

            // 投影比例（0 = 双眼中点，1 = 嘴中点），允许侧脸带来的延长
            val t = ((nx - eyeMidX) * segX + (ny - eyeMidY) * segY) / segLen2
            if (t < -0.5f || t > 1.5f) return false

            return true
        }

        /** 由原始浮点数组构造（内部使用） */
        internal fun of(
            left: Float, top: Float, right: Float, bottom: Float,
            score: Float, landmarks: FloatArray?,
        ): Face {
            val l = min(left, right); val t = min(top, bottom)
            val r = max(left, right); val b = max(top, bottom)
            return Face(RectF(l, t, r, b), score, landmarks)
        }
    }
}
