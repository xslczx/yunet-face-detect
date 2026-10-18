package com.yunet.face

import java.nio.ByteBuffer

/** 模型来源类型 */
enum class ModelSourceType { ASSET, FILE, BYTES }

/**
 * 输入通道顺序。
 * YuNet 官方训练/评测管线为 BGR（OpenCV 读取顺序，to_rgb=False）。
 * litert-community 的 yunet_fp16.tflite 模型卡同样标注 BGR、0-255、不做归一化。
 */
enum class ChannelOrder { BGR, RGB }

/**
 * 几何缩放模式：
 * - [LETTERBOX]：等比缩放 + 右下补边（padValue），官方评测管线（build_eval_transforms）；
 * - [STRETCH]：直接拉伸到输入尺寸，官方训练管线（build_train_transforms）。
 * 两种模式均可正确映射回原图坐标；档案照裁剪场景可对比两者检出率后选择。
 */
enum class ResizeMode { LETTERBOX, STRETCH }

/** 推理委托。GPU 需要额外引入 org.tensorflow:tensorflow-lite-gpu，否则自动回退 CPU。 */
enum class DelegateType { CPU, GPU, NNAPI }

/** 输入张量布局。AUTO 按模型实际形状推断（[1,3,H,W] 或 [1,H,W,3]）。 */
enum class InputLayout { AUTO, NCHW, NHWC }

/**
 * 12 路原始输出的张量索引映射（cls/obj/bbox/kps × 各 stride）。
 *
 * 默认值来自 libfacedetection.train 官方导出顺序：
 * cls_8, cls_16, cls_32, obj_8, obj_16, obj_32, bbox_8, bbox_16, bbox_32, kps_8, kps_16, kps_32
 *
 * 若你的模型输出顺序不同（或 strides 数量不为 3），请显式构造并设置。
 */
class OutputIndices(
    val cls: IntArray,
    val obj: IntArray,
    val bbox: IntArray,
    val kps: IntArray,
) {
    companion object {
        /** 3 stride（8/16/32）默认映射 */
        @JvmStatic
        fun default(): OutputIndices = OutputIndices(
            cls = intArrayOf(0, 1, 2),
            obj = intArrayOf(3, 4, 5),
            bbox = intArrayOf(6, 7, 8),
            kps = intArrayOf(9, 10, 11),
        )
    }
}

/**
 * YuNet 人脸检测库配置。使用 [Builder] 构造，所有参数均可选（有默认值）。
 *
 * 默认参数对应 litert-community 的 yunet_fp16.tflite
 * （libfacedetection.train 导出、LiteRT fp16 转换的社区模型）。
 */
class YuNetConfig internal constructor(
    // ---- 模型来源（三选一） ----
    val sourceType: ModelSourceType,
    val assetName: String?,
    val filePath: String?,
    val modelBytes: ByteBuffer?,

    // ---- 输入 ----
    /** 0 表示按模型实际形状自适应 */
    val inputWidth: Int,
    val inputHeight: Int,
    val inputLayout: InputLayout,
    val channelOrder: ChannelOrder,
    val resizeMode: ResizeMode,
    /** letterbox 补边灰度值 0-255 */
    val padValue: Int,
    /** 像素归一化：out = pixel * normalizeScale + normalizeMean。官方为 1/0（即原始 0-255） */
    val normalizeScale: Float,
    val normalizeMean: Float,
    /** 送入推理前将输入位图最大边限制到该值（0 表示不限制），防止超大图 OOM */
    val maxInputImageDim: Int,

    // ---- 解码 / 后处理 ----
    val scoreThreshold: Float,
    val nmsThreshold: Float,
    /** NMS 后最多保留的人脸数，0 表示不限制 */
    val topK: Int,
    /** NMS 前按分数保留的候选上限（0 表示不限制），用于约束 O(n^2) 最坏耗时 */
    val nmsPreTopK: Int,
    val strides: IntArray,
    /** cls/obj 输出是否未经 sigmoid（原始 logits）。tflite 已内置 sigmoid，默认 false */
    val applySigmoidToScores: Boolean,
    /** 自定义 12 路输出索引映射；null 用默认顺序（或按张量名自动识别） */
    val outputIndices: OutputIndices?,

    // ---- 运行时 ----
    val delegate: DelegateType,
    val numThreads: Int,
    /** 委托初始化失败时是否回退 CPU */
    val fallbackToCpu: Boolean,
    /** detectFile 是否按 EXIF 方向自动转正 */
    val autoRotateExif: Boolean,
    /** 异步回调是否切回主线程 */
    val callbacksOnMainThread: Boolean,
) {
    internal fun stridesCopy(): IntArray = strides.copyOf()

    class Builder {
        private var sourceType: ModelSourceType? = null
        private var assetName: String? = null
        private var filePath: String? = null
        private var modelBytes: ByteBuffer? = null
        private var inputWidth = 0
        private var inputHeight = 0
        private var inputLayout = InputLayout.AUTO
        private var channelOrder = ChannelOrder.BGR
        private var resizeMode = ResizeMode.LETTERBOX
        private var padValue = 0
        private var normalizeScale = 1f
        private var normalizeMean = 0f
        private var maxInputImageDim = 2048
        private var scoreThreshold = 0.6f
        private var nmsThreshold = 0.45f
        private var topK = 0
        private var nmsPreTopK = 5000
        private var strides = intArrayOf(8, 16, 32)
        private var applySigmoidToScores = false
        private var outputIndices: OutputIndices? = null
        private var delegate = DelegateType.CPU
        private var numThreads = 4
        private var fallbackToCpu = true
        private var autoRotateExif = true
        private var callbacksOnMainThread = true

        /** 从 assets 加载模型，如 "yunet_fp16.tflite" */
        fun useAsset(name: String) = apply {
            sourceType = ModelSourceType.ASSET; assetName = name; filePath = null; modelBytes = null
        }

        /** 从文件路径加载模型 */
        fun useFile(path: String) = apply {
            sourceType = ModelSourceType.FILE; filePath = path; assetName = null; modelBytes = null
        }

        /** 从内存中的模型字节加载（position..limit 需为完整模型文件） */
        fun useBytes(buffer: ByteBuffer) = apply {
            sourceType = ModelSourceType.BYTES; modelBytes = buffer; assetName = null; filePath = null
        }

        /** 指定网络输入尺寸；0（默认）表示按模型实际形状自适应 */
        fun inputSize(width: Int, height: Int) = apply { inputWidth = width; inputHeight = height }

        fun inputLayout(layout: InputLayout) = apply { inputLayout = layout }
        fun channelOrder(order: ChannelOrder) = apply { channelOrder = order }
        fun resizeMode(mode: ResizeMode) = apply { resizeMode = mode }
        fun padValue(value: Int) = apply { padValue = value }

        /** 自定义像素归一化，如 normalize(1f / 255f, 0f) */
        fun normalize(scale: Float, mean: Float) = apply { normalizeScale = scale; normalizeMean = mean }

        fun maxInputImageDim(dim: Int) = apply { maxInputImageDim = dim }

        /** 人脸分数阈值。官方评测默认 0.02（mAP 评估）；应用建议 0.3 ~ 0.9 */
        fun scoreThreshold(value: Float) = apply { scoreThreshold = value }

        /** NMS IoU 阈值。官方默认 0.45 */
        fun nmsThreshold(value: Float) = apply { nmsThreshold = value }

        fun topK(value: Int) = apply { topK = value }
        fun nmsPreTopK(value: Int) = apply { nmsPreTopK = value }

        /** 自定义 strides（需同时通过 outputIndices 指定输出映射） */
        fun strides(vararg values: Int) = apply {
            require(values.isNotEmpty()) { "strides 不能为空" }
            strides = values.copyOf().sortedArray()
        }

        fun applySigmoidToScores(enabled: Boolean) = apply { applySigmoidToScores = enabled }
        fun outputIndices(indices: OutputIndices) = apply { outputIndices = indices }
        fun delegate(type: DelegateType) = apply { delegate = type }
        fun numThreads(threads: Int) = apply { numThreads = threads }
        fun fallbackToCpu(enabled: Boolean) = apply { fallbackToCpu = enabled }
        fun autoRotateExif(enabled: Boolean) = apply { autoRotateExif = enabled }
        fun callbacksOnMainThread(enabled: Boolean) = apply { callbacksOnMainThread = enabled }

        @Throws(IllegalArgumentException::class)
        fun build(): YuNetConfig {
            val st = sourceType ?: throw IllegalArgumentException("必须指定模型来源：useAsset / useFile / useBytes")
            if (inputWidth < 0 || inputHeight < 0) throw IllegalArgumentException("inputSize 不能为负")
            if (inputWidth > 0 && (inputWidth != inputHeight || inputWidth % 32 != 0))
                throw IllegalArgumentException("inputWidth/inputHeight 需相等且为 32 的倍数，当前 ${inputWidth}x${inputHeight}")
            if (scoreThreshold < 0f || scoreThreshold > 1f) throw IllegalArgumentException("scoreThreshold 应在 [0,1]，当前 $scoreThreshold")
            if (nmsThreshold <= 0f || nmsThreshold >= 1f) throw IllegalArgumentException("nmsThreshold 应在 (0,1)，当前 $nmsThreshold")
            if (topK < 0) throw IllegalArgumentException("topK 不能为负")
            if (nmsPreTopK < 0) throw IllegalArgumentException("nmsPreTopK 不能为负")
            if (strides.isEmpty() || strides.any { it <= 0 }) throw IllegalArgumentException("strides 非法: ${strides.toList()}")
            if (strides.toSet().size != strides.size) throw IllegalArgumentException("strides 含重复值: ${strides.toList()}")
            if (numThreads < 0) throw IllegalArgumentException("numThreads 不能为负")
            if (padValue < 0 || padValue > 255) throw IllegalArgumentException("padValue 应在 [0,255]，当前 $padValue")
            if (normalizeScale <= 0f) throw IllegalArgumentException("normalizeScale 需 > 0")
            if (maxInputImageDim in 1..319) throw IllegalArgumentException("maxInputImageDim 应 >= 320 或为 0（不限制）")
            return YuNetConfig(
                sourceType = st,
                assetName = assetName,
                filePath = filePath,
                modelBytes = modelBytes,
                inputWidth = inputWidth,
                inputHeight = inputHeight,
                inputLayout = inputLayout,
                channelOrder = channelOrder,
                resizeMode = resizeMode,
                padValue = padValue,
                normalizeScale = normalizeScale,
                normalizeMean = normalizeMean,
                maxInputImageDim = maxInputImageDim,
                scoreThreshold = scoreThreshold,
                nmsThreshold = nmsThreshold,
                topK = topK,
                nmsPreTopK = nmsPreTopK,
                strides = strides,
                applySigmoidToScores = applySigmoidToScores,
                outputIndices = outputIndices,
                delegate = delegate,
                numThreads = numThreads,
                fallbackToCpu = fallbackToCpu,
                autoRotateExif = autoRotateExif,
                callbacksOnMainThread = callbacksOnMainThread,
            )
        }
    }
}
