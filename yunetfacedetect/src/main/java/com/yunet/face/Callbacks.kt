package com.yunet.face

import android.graphics.Bitmap

/** 异步检测回调。回调线程由配置 callbacksOnMainThread 决定（默认主线程）。 */
interface DetectCallback {
    /** 检测完成。[faces] 按分数降序，可能为空列表。 */
    fun onSuccess(faces: List<Face>)

    /** 检测失败。[error] 为 [YuNetException] 子类，不会为 null。 */
    fun onError(error: YuNetException)
}

/** 异步创建检测器回调 */
interface CreateCallback {
    fun onCreated(detector: YuNetDetector)
    fun onError(error: YuNetException)
}

/**
 * 模型元信息（加载时自省结果）。用于校验模型结构、排查输出映射问题。
 */
class ModelInfo(
    val modelSizeBytes: Long,
    /** 输入张量形状，如 [1, 3, 640, 640] */
    val inputShape: IntArray,
    val inputLayout: InputLayout,
    /** "FLOAT32" / "FLOAT16" */
    val inputDtype: String,
    val inputChannels: Int,
    val inputWidth: Int,
    val inputHeight: Int,
    val outputCount: Int,
    val outputNames: Array<String>,
    val outputShapes: Array<IntArray>,
    val outputDtypes: Array<String>,
    /** 每个输出张量解析出的角色，如 "cls(stride=8)"；无法识别为 "?" */
    val outputRoles: Array<String>,
    val strides: IntArray,
    /** 每个 stride 的 anchor 数量，如 [6400, 1600, 400] */
    val anchorCounts: IntArray,
    /** 输出角色是否依据张量名推断（false 表示使用默认/显式索引顺序） */
    val mappingFromNames: Boolean,
    val initMs: Long,
) {
    val totalAnchors: Int get() = anchorCounts.sum()

    override fun toString(): String = buildString {
        appendLine("ModelInfo(")
        appendLine("  modelSizeBytes=$modelSizeBytes")
        appendLine("  input=${inputShape.toList()} $inputLayout $inputDtype (${inputWidth}x${inputHeight}, ch=$inputChannels)")
        appendLine("  outputs=$outputCount:")
        for (i in 0 until outputCount) {
            appendLine("    [$i] ${outputNames[i]} shape=${outputShapes[i].toList()} ${outputDtypes[i]} -> ${outputRoles[i]}")
        }
        appendLine("  strides=${strides.toList()} anchors=${anchorCounts.toList()} total=$totalAnchors")
        appendLine("  mappingFromNames=$mappingFromNames initMs=$initMs")
        append(")")
    }
}

/**
 * 人脸检测器抽象接口。后续接入其他模型（RetinaFace / SCRFD 等）时实现该接口即可，
 * 调用方代码无需改动。
 */
interface FaceDetector {
    /**
     * 同步检测。会阻塞调用线程，勿在主线程调用。
     * 失败时抛出 [YuNetException] 子类。
     */
    @Throws(YuNetException::class)
    fun detectSync(bitmap: Bitmap): List<Face>

    /**
     * 异步检测。不抛异常，结果或错误经 [callback] 返回。
     * 内部单线程串行执行；同一检测器实例的多次调用会排队。
     */
    fun detectAsync(bitmap: Bitmap, callback: DetectCallback)

    /** 释放模型与线程资源。幂等；关闭后继续调用会抛 [YuNetClosedException]。 */
    fun close()
}
