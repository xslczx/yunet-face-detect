package com.yunet.face

/**
 * `detectFileSyncDetailed` 的结构化结果。
 *
 * 坐标契约（容易弄混，务必区分两组坐标）：
 * - [faces]：**文件存储像素**坐标系，基于（[fileWidth] × [fileHeight]）。
 *   即不施加任何 EXIF 变换时的坐标，与 [YuNetDetector.detectFileSync] 的既有语义一致。
 *   适合与“文件原始字节流”相关的消费方（如自绘 EXIF 后处理）。
 * - [uprightFaces]：**EXIF 转正后**坐标系，基于（[uprightWidth] × [uprightHeight]）。
 *   即“用系统图库正常查看该照片”时看到的图像坐标。
 *   画到转正后的 ImageView / 用于裁剪定位时，请使用这一组。
 *
 * 镜像照片（orientation 2/4/5/7，常见于前摄）两组坐标不同，且 [faces] 的
 * 左时针/左右眼命名相对 [uprightFaces] 是镜像对称的。
 */
class FileDetectionResult(
    /** EXIF 转正所需的顺时针旋转角度（0/90/180/270） */
    val rotationDegrees: Int,
    /** 转正前是否需要水平镜像 */
    val mirrored: Boolean,
    /** 文件存储像素尺寸（[faces] 所在坐标系） */
    val fileWidth: Int,
    val fileHeight: Int,
    /** 文件存储像素坐标系下的人脸（按分数降序） */
    val faces: List<Face>,
    /** EXIF 转正后坐标系下的人脸（按分数降序，与 [faces] 按索引一一对应） */
    val uprightFaces: List<Face>,
) {

    /** 转正后的图像宽度（[uprightFaces] 所在坐标系） */
    val uprightWidth: Int =
        if (rotationDegrees == 90 || rotationDegrees == 270) fileHeight else fileWidth

    /** 转正后的图像高度 */
    val uprightHeight: Int =
        if (rotationDegrees == 90 || rotationDegrees == 270) fileWidth else fileHeight

    override fun toString(): String =
        "FileDetectionResult(size=${fileWidth}x${fileHeight}, rotation=$rotationDegrees, " +
            "mirrored=$mirrored, upright=${uprightWidth}x${uprightHeight}, faces=${faces.size})"
}
