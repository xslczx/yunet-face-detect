package com.yunet.face

import android.content.Context
import android.util.Log
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Delegate
import org.tensorflow.lite.Interpreter
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

/**
 * TFLite/LiteRT 引擎封装：模型加载、结构自省、输入填充、推理、资源释放。
 *
 * 设计要点：
 * - 仅支持 FLOAT32 I/O（LiteRT/TFLite Java API 的 DataType 枚举已不含 FLOAT16；
 *   fp16 权重模型的输入输出通常仍为 FLOAT32，与 litert-community 模型卡的 writeFloat/readFloat 用法一致）；
 * - 输出张量角色优先按张量名（cls_8/obj_16 等）识别，失败时回退默认/显式索引顺序；
 * - GPU 委托通过反射加载，未引入 tensorflow-lite-gpu 时按配置回退 CPU；
 * - 所有推理在锁内串行执行（TFLite Interpreter 非线程安全）。
 */
internal class TfliteEngine(private val config: YuNetConfig) {

    internal class OutputSpec(
        val index: Int,
        val name: String,
        val shape: IntArray,
        val channels: Int,
        val numElements: Int,
        val dtype: DataType,
    )

    private val lock = Any()
    private var interpreter: Interpreter? = null
    private var tempModelFile: File? = null
    private var inputBuffer: ByteBuffer? = null
    private var outputSpecs: List<OutputSpec> = emptyList()

    /** 每个输出张量一个 direct ByteBuffer（与 [outputSpecs] 同序），跨次推理复用 */
    private var outputBuffers: List<ByteBuffer> = emptyList()
    private var closed = false

    var modelInfo: ModelInfo? = null
        private set

    /** 自省最终确定的输出索引映射（可能与配置不同：名称识别优先） */
    var resolvedIndices: OutputIndices? = null
        private set

    fun open(context: Context): ModelInfo {
        synchronized(lock) {
            if (closed) throw YuNetClosedException("引擎已关闭")
            val t0 = System.currentTimeMillis()
            val buffer = loadModelBuffer(context)
            val options = buildOptions()
            val interp = try {
                Interpreter(buffer, options)
            } catch (t: Throwable) {
                throw YuNetModelException("TFLite 模型加载失败: ${t.message}", t)
            }
            interpreter = interp
            val info = introspect(interp, System.currentTimeMillis() - t0)
            modelInfo = info
            return info
        }
    }

    // ---------------- 模型加载 ----------------

    private fun loadModelBuffer(context: Context): ByteBuffer {
        return when (config.sourceType) {
            ModelSourceType.ASSET -> {
                val name = config.assetName
                    ?: throw YuNetModelException("模型来源为 ASSET 但 assetName 为空")
                val tmp = File(context.cacheDir, "yunet_model_${System.nanoTime()}.tflite")
                try {
                    context.assets.open(name).use { input ->
                        FileOutputStream(tmp).use { out -> input.copyTo(out) }
                    }
                } catch (t: Throwable) {
                    runCatching { tmp.delete() }
                    throw YuNetModelException("读取 assets 模型 '$name' 失败: ${t.message}", t)
                }
                tempModelFile = tmp
                tmp.toMappedBuffer()
            }
            ModelSourceType.FILE -> {
                val f = File(config.filePath ?: throw YuNetModelException("模型来源为 FILE 但 filePath 为空"))
                when {
                    !f.exists() || !f.isFile -> throw YuNetModelException("模型文件不存在: ${f.absolutePath}")
                    !f.canRead() -> throw YuNetModelException("模型文件不可读: ${f.absolutePath}")
                    f.length() <= 0L -> throw YuNetModelException("模型文件为空: ${f.absolutePath}")
                }
                f.toMappedBuffer()
            }
            ModelSourceType.BYTES -> {
                val b = config.modelBytes ?: throw YuNetModelException("模型来源为 BYTES 但 modelBytes 为空")
                // 约定：position..limit 需包含完整模型文件
                b.duplicate().order(ByteOrder.nativeOrder())
            }
        }
    }

    private fun File.toMappedBuffer(): ByteBuffer =
        FileInputStream(this).use { fis ->
            fis.channel.use { ch -> ch.map(FileChannel.MapMode.READ_ONLY, 0, ch.size()) }
        }

    @Suppress("DEPRECATION")
    private fun buildOptions(): Interpreter.Options {
        val options = Interpreter.Options()
        if (config.numThreads > 0) {
            options.setNumThreads(config.numThreads)
        }
        when (config.delegate) {
            DelegateType.CPU -> Unit
            DelegateType.GPU -> {
                val delegate: Delegate? = try {
                    val cls = Class.forName("org.tensorflow.lite.gpu.GpuDelegate")
                    cls.getDeclaredConstructor().newInstance() as Delegate
                } catch (t: Throwable) {
                    Log.w(TAG, "GpuDelegate 反射失败: ${t.message}")
                    null
                }
                if (delegate != null) {
                    options.addDelegate(delegate)
                } else if (config.fallbackToCpu) {
                    Log.w(TAG, "GPU 委托不可用（未引入 tensorflow-lite-gpu 或设备不支持），已回退 CPU")
                } else {
                    throw YuNetModelException(
                        "GPU 委托不可用：请添加依赖 org.tensorflow:tensorflow-lite-gpu，或设置 fallbackToCpu=true"
                    )
                }
            }
            DelegateType.NNAPI -> {
                try {
                    options.setUseNNAPI(true)
                } catch (t: Throwable) {
                    if (!config.fallbackToCpu) {
                        throw YuNetModelException("NNAPI 不可用: ${t.message}", t)
                    }
                    Log.w(TAG, "NNAPI 不可用，已回退 CPU")
                }
            }
        }
        return options
    }

    // ---------------- 结构自省 ----------------

    private fun introspect(interp: Interpreter, initMs: Long): ModelInfo {
        val inTensor = try {
            interp.getInputTensor(0)
        } catch (t: Throwable) {
            throw YuNetIncompatibleModelException("无法读取输入张量 0: ${t.message}", t)
        }
        val inShape = inTensor.shape()
        if (inShape.size != 4) {
            throw YuNetIncompatibleModelException("输入张量 rank=${inShape.size}，期望 4 维 [1,C,H,W] 或 [1,H,W,C]")
        }
        val layout = when (config.inputLayout) {
            InputLayout.NCHW -> InputLayout.NCHW
            InputLayout.NHWC -> InputLayout.NHWC
            InputLayout.AUTO -> if (inShape[3] in 1..4) InputLayout.NHWC else InputLayout.NCHW
        }
        val channels = if (layout == InputLayout.NCHW) inShape[1] else inShape[3]
        val inH = if (layout == InputLayout.NCHW) inShape[2] else inShape[1]
        val inW = if (layout == InputLayout.NCHW) inShape[3] else inShape[2]
        if (channels != 1 && channels != 3) {
            throw YuNetIncompatibleModelException("输入通道数=$channels，期望 1 或 3")
        }
        if (inW <= 0 || inH <= 0) throw YuNetIncompatibleModelException("输入尺寸非法: ${inW}x${inH}")
        if (config.inputWidth > 0 && (config.inputWidth != inW || config.inputHeight != inH)) {
            throw YuNetIncompatibleModelException(
                "配置输入尺寸 ${config.inputWidth}x${config.inputHeight} 与模型实际 ${inW}x${inH} 不一致"
            )
        }
        if (inW % 32 != 0 || inH % 32 != 0) {
            throw YuNetIncompatibleModelException("输入尺寸 ${inW}x${inH} 需为 32 的倍数（strides=8/16/32）")
        }
        if (inTensor.dataType() != DataType.FLOAT32) {
            throw YuNetIncompatibleModelException(
                "输入 dtype=${inTensor.dataType()}，本库仅支持 FLOAT32。" +
                    "（LiteRT/TFLite Java API 的 DataType 枚举不含 FLOAT16；fp16 权重模型的 I/O 通常仍为 FLOAT32）"
            )
        }

        val outCount = try {
            interp.outputTensorCount
        } catch (t: Throwable) {
            0
        }
        if (outCount <= 0) throw YuNetIncompatibleModelException("模型无输出张量")
        val specs = ArrayList<OutputSpec>(outCount)
        for (i in 0 until outCount) {
            val t = try {
                interp.getOutputTensor(i)
            } catch (e: Throwable) {
                throw YuNetIncompatibleModelException("无法读取输出张量 $i: ${e.message}", e)
            }
            val shape = t.shape()
            if (shape.isEmpty() || shape.any { it <= 0 }) {
                throw YuNetIncompatibleModelException(
                    "输出张量 $i 形状非法或为动态形状: ${shape.toList()}（本库要求固定形状模型）"
                )
            }
            val c = shape.last()
            if (c != 1 && c != 4 && c != 10) {
                throw YuNetIncompatibleModelException(
                    "输出张量 $i 末维=$c，不属于 YuNet 的 cls/obj(C=1) / bbox(C=4) / kps(C=10) 结构"
                )
            }
            if (t.dataType() != DataType.FLOAT32) {
                throw YuNetIncompatibleModelException(
                    "输出张量 $i dtype=${t.dataType()}，本库仅支持 FLOAT32"
                )
            }
            val numElements = t.numElements()
            specs.add(OutputSpec(i, t.name() ?: "output_$i", shape, c, numElements, t.dataType()))
        }

        val strides = config.stridesCopy()
        val strideCount = strides.size
        val ones = specs.count { it.channels == 1 }
        val fours = specs.count { it.channels == 4 }
        val tens = specs.count { it.channels == 10 }
        if (ones != strideCount * 2 || fours != strideCount || tens != strideCount) {
            throw YuNetIncompatibleModelException(
                "输出结构不符：期望 cls+obj=${strideCount * 2} 个 C=1、bbox=$strideCount 个 C=4、kps=$strideCount 个 C=10；" +
                    "实际 C1=$ones, C4=$fours, C10=$tens。本库面向 libfacedetection.train 导出的 12 路 YuNet 模型"
            )
        }

        // 角色分配：优先按张量名（转换工具通常保留 cls_8/obj_16 等名字）
        val parsed = specs.mapNotNull { s ->
            parseTensorName(s.name)?.let { (role, st) -> s to (role to st) }
        }
        val useNames = parsed.size == specs.size &&
            parsed.all { (_, pair) -> strides.contains(pair.second) } &&
            parsed.groupBy { it.second.first }.entries.let { byRole ->
                byRole.size == 4 && byRole.all { it.value.size == strideCount } &&
                    byRole.all { entry -> entry.value.map { it.second.second }.toSet().size == strideCount }
            }

        val clsIdx = IntArray(strideCount) { -1 }
        val objIdx = IntArray(strideCount) { -1 }
        val bboxIdx = IntArray(strideCount) { -1 }
        val kpsIdx = IntArray(strideCount) { -1 }
        if (useNames) {
            for ((spec, pair) in parsed) {
                val si = strides.indexOf(pair.second)
                when (pair.first) {
                    "cls" -> clsIdx[si] = spec.index
                    "obj" -> objIdx[si] = spec.index
                    "bbox" -> bboxIdx[si] = spec.index
                    "kps" -> kpsIdx[si] = spec.index
                }
            }
            resolvedIndices = OutputIndices(clsIdx, objIdx, bboxIdx, kpsIdx)
        } else {
            val idx = config.outputIndices ?: run {
                if (strideCount != 3) {
                    throw YuNetIncompatibleModelException(
                        "strides 数量=$strideCount ≠ 3 时必须通过 outputIndices 显式指定输出映射"
                    )
                }
                OutputIndices.default()
            }
            require(
                idx.cls.size == strideCount && idx.obj.size == strideCount &&
                    idx.bbox.size == strideCount && idx.kps.size == strideCount
            ) { "outputIndices 数组长度需与 strides 数量一致" }
            for (si in 0 until strideCount) {
                clsIdx[si] = idx.cls[si]
                objIdx[si] = idx.obj[si]
                bboxIdx[si] = idx.bbox[si]
                kpsIdx[si] = idx.kps[si]
            }
            resolvedIndices = idx
        }

        val allIdx = clsIdx + objIdx + bboxIdx + kpsIdx
        if (allIdx.any { it < 0 || it >= outCount }) {
            throw YuNetIncompatibleModelException("输出索引越界: ${allIdx.toList()}")
        }
        if (allIdx.toSet().size != allIdx.size) {
            throw YuNetIncompatibleModelException("输出索引重复: ${allIdx.toList()}")
        }

        // anchor 数校验：同一 stride 的 4 个分支 N 需一致且等于特征图尺寸
        val anchorCounts = IntArray(strideCount)
        for (si in 0 until strideCount) {
            val clsS = specs[clsIdx[si]]
            val objS = specs[objIdx[si]]
            val bboxS = specs[bboxIdx[si]]
            val kpsS = specs[kpsIdx[si]]
            val nCls = clsS.numElements / clsS.channels
            val nObj = objS.numElements / objS.channels
            val nBbox = bboxS.numElements / bboxS.channels
            val nKps = kpsS.numElements / kpsS.channels
            if (nCls != nObj || nCls != nBbox || nCls != nKps) {
                throw YuNetIncompatibleModelException(
                    "stride=${strides[si]} 各分支 anchor 数不一致: cls=$nCls, obj=$nObj, bbox=$nBbox, kps=$nKps"
                )
            }
            val s = strides[si]
            val fmW = inW / s
            val fmH = inH / s
            val expected = fmW * fmH
            if (expected != nCls) {
                throw YuNetIncompatibleModelException(
                    "stride=$s anchor 数不匹配: 模型输出 $nCls，按 ${fmW}x${fmH} 期望 $expected；" +
                        "请检查配置 inputSize / strides 与模型是否一致"
                )
            }
            anchorCounts[si] = nCls
        }

        val roles = Array(outCount) { "?" }
        for (si in 0 until strideCount) {
            roles[clsIdx[si]] = "cls(stride=${strides[si]})"
            roles[objIdx[si]] = "obj(stride=${strides[si]})"
            roles[bboxIdx[si]] = "bbox(stride=${strides[si]})"
            roles[kpsIdx[si]] = "kps(stride=${strides[si]})"
        }

        outputSpecs = specs.sortedBy { it.index }

        val modelSize = when (config.sourceType) {
            ModelSourceType.FILE -> File(config.filePath ?: "").length()
            ModelSourceType.BYTES -> (config.modelBytes?.remaining() ?: 0).toLong()
            ModelSourceType.ASSET -> -1L
        }

        return ModelInfo(
            modelSizeBytes = modelSize,
            inputShape = inShape,
            inputLayout = layout,
            inputDtype = inTensor.dataType().name,
            inputChannels = channels,
            inputWidth = inW,
            inputHeight = inH,
            outputCount = outCount,
            outputNames = specs.sortedBy { it.index }.map { it.name }.toTypedArray(),
            outputShapes = specs.sortedBy { it.index }.map { it.shape }.toTypedArray(),
            outputDtypes = specs.sortedBy { it.index }.map { it.dtype.name }.toTypedArray(),
            outputRoles = roles,
            strides = strides,
            anchorCounts = anchorCounts,
            mappingFromNames = useNames,
            initMs = initMs,
        )
    }

    /** 从张量名解析角色与 stride，如 "cls_8" -> ("cls", 8) */
    private fun parseTensorName(name: String): Pair<String, Int>? {
        val lower = name.lowercase()
        val role = when {
            lower.contains("cls") || lower.contains("class") -> "cls"
            lower.contains("obj") || lower.contains("objectness") -> "obj"
            lower.contains("bbox") || lower.contains("box") -> "bbox"
            lower.contains("kps") || lower.contains("keypoint") || lower.contains("landmark") -> "kps"
            else -> return null
        }
        val stride = Regex("(\\d+)").findAll(name)
            .mapNotNull { it.value.toIntOrNull() }
            .lastOrNull() ?: return null
        return role to stride
    }

    // ---------------- 推理 ----------------

    /**
     * 执行一次推理。
     * @param pixels 按 [ModelInfo.inputLayout] 与通道序排列的像素值（长度 = C*H*W）
     * @return 各输出张量平铺 FloatArray，按张量索引顺序
     */
    fun run(pixels: FloatArray): Array<FloatArray> {
        synchronized(lock) {
            if (closed) throw YuNetClosedException("引擎已关闭")
            val interp = interpreter ?: throw YuNetClosedException("引擎未初始化")
            val inTensor = try {
                interp.getInputTensor(0)
            } catch (t: Throwable) {
                throw YuNetInferenceException("读取输入张量失败: ${t.message}", t)
            }
            val expectedElements = inTensor.numElements()
            if (pixels.size != expectedElements) {
                throw YuNetInputException(
                    "像素数据长度 ${pixels.size} 与模型输入不符（期望 $expectedElements 个元素）"
                )
            }
            val buf = inputBuffer
                ?: ByteBuffer.allocateDirect(expectedElements * 4).order(ByteOrder.nativeOrder())
                    .also { inputBuffer = it }
            buf.clear()
            for (v in pixels) buf.putFloat(v)
            buf.rewind()

            val specs = outputSpecs
            if (specs.isEmpty()) throw YuNetIncompatibleModelException("引擎未完成结构自省")

            // 输出容器用 direct ByteBuffer 而不是 float 数组：
            // LiteRT 1.x 的 Tensor.copyTo 会校验数组的「形状」是否与张量一致
            // （如 [1,6400,1] 要求 float[][][]），扁平 FloatArray 会抛
            // IllegalArgumentException: ... incompatible shape。
            // ByteBuffer 只校验字节容量，且可跨次推理复用。
            if (outputBuffers.size != specs.size) {
                outputBuffers = specs.map {
                    ByteBuffer.allocateDirect(it.numElements * 4).order(ByteOrder.nativeOrder())
                }
            }
            val outputs = HashMap<Int, Any>(specs.size)
            specs.forEachIndexed { i, s ->
                val b = outputBuffers[i]
                b.clear()
                outputs[s.index] = b
            }
            try {
                // 必须用 runForMultipleInputsOutputs，不能用 run(input, output)：
                // LiteRT 1.x 的 Interpreter.run(Object, Object) 把第二个参数当作「单个输出张量的目标容器」，
                // 传 Map 进去会报 DataType error: cannot resolve DataType of java.util.HashMap
                //（tensorflow-lite:2.17.0 已重定向到 com.google.ai.edge.litert:litert，实测即此行为）。
                runMultipleOutputs(interp, buf, outputs)
            } catch (t: Throwable) {
                throw YuNetInferenceException("TFLite 推理失败: ${t.message}", t)
            }
            return specs.mapIndexed { i, s ->
                val f = FloatArray(s.numElements)
                outputBuffers[i].rewind()
                outputBuffers[i].asFloatBuffer().get(f)
                f
            }.toTypedArray()
        }
    }

    @Suppress("DEPRECATION")
    private fun runMultipleOutputs(interp: Interpreter, input: ByteBuffer, outputs: Map<Int, Any>) {
        interp.runForMultipleInputsOutputs(arrayOf<Any>(input), outputs)
    }

    fun close() {
        synchronized(lock) {
            if (closed) return
            closed = true
            runCatching { interpreter?.close() }
            interpreter = null
            inputBuffer = null
            outputBuffers = emptyList()
            outputSpecs = emptyList()
            runCatching { tempModelFile?.delete() }
            tempModelFile = null
        }
    }

    companion object {
        private const val TAG = "TfliteEngine"
    }
}
