package com.yunet.face

/**
 * EXIF 方向的完整几何语义（旋转 + 镜像）。
 *
 * androidx `ExifInterface` 的官方约定：把存储像素变为“转正（可正常展示）”图像的变换是
 * **先水平翻转（[isFlipped] 为 true 时），再顺时针旋转 [rotationDegrees]**。
 * 注意镜像与旋转不可交换（对 90/270），必须保持该顺序：
 * - orientation 2：仅水平镜像（rotation=0）
 * - orientation 4：水平镜像后旋转 180°（等效垂直翻转）
 * - orientation 5：水平镜像后旋转 270°（等效主对角线转置）
 * - orientation 7：水平镜像后旋转 90°（等效反对角线翻转）
 *
 * 前摄/美颜相机的镜像照片（orientation 2/4 最常见）依赖此语义才能正确检测与定位。
 */
class ExifTransform(
    /** 转正所需的顺时针旋转角度（0/90/180/270） */
    val rotationDegrees: Int,
    /** 是否需要在旋转前先对存储像素水平镜像（EXIF orientation 2/4/5/7） */
    val mirrored: Boolean,
) {

    /** 转正后的图像尺寸（宽, 高）：rotation 为 90/270 时宽高互换 */
    fun uprightSize(fileWidth: Int, fileHeight: Int): IntArray =
        if (rotationDegrees == 90 || rotationDegrees == 270) {
            intArrayOf(fileHeight, fileWidth)
        } else {
            intArrayOf(fileWidth, fileHeight)
        }

    override fun toString(): String = "ExifTransform(rotation=$rotationDegrees, mirrored=$mirrored)"

    companion object {
        /** 无 EXIF 变换。App 内自行解码的 Bitmap 通常已由系统转正。 */
        @JvmField
        val IDENTITY = ExifTransform(0, false)

        /** 由 ExifInterface 的读取结果构造；角度归一化到 0/90/180/270 */
        @JvmStatic
        fun of(rotationDegrees: Int, isFlipped: Boolean): ExifTransform {
            val r = ((rotationDegrees % 360) + 360) % 360
            val deg = when {
                r < 45 || r >= 315 -> 0
                r < 135 -> 90
                r < 225 -> 180
                else -> 270
            }
            return ExifTransform(deg, isFlipped)
        }
    }
}
