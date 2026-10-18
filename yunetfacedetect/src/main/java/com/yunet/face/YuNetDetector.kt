package com.yunet.face

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Handler
import android.os.Looper
import androidx.exifinterface.media.ExifInterface
import com.yunet.face.decode.OutputDecoder
import com.yunet.face.decode.RawFace
import com.yunet.face.decode.RawYuNetDecoder
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.util.concurrent.Executors
import kotlin.math.max

/**
 * YuNet 人脸检测器主入口。
 *
 * 用法（Kotlin）：
 * ```
 * val detector = YuNetDetector.create(context, YuNetConfig.Builder()
 *     .useAsset("yunet_fp16.tflite")
 *     .scoreThreshold(0.6f)
 *     .build())
 * val faces = detector.detectSync(bitmap)          // 同步（阻塞）
 * detector.detectAsync(bitmap) { faces, err -> }   // 异步回调
 * val faces2 = detector.detect(bitmap)             // 协程 suspend
 * detector.detectFileSync("/path/photo.jpg")       // 文件 + EXIF 自动转正
 * detector.close()
 * ```
 *
 * 线程模型：同步方法由调用方保证不在主线程执行；异步方法内部单线程串行，
 * 同一实例的多次异步调用按提交顺序排队。TFLite Interpreter 非线程安全，
 * 全部推理在引擎锁内执行。
 */
class YuNetDetector private constructor(
    private val config: YuNetConfig,
    private val engine: TfliteEngine,
    private val preprocessor: Preprocessor,
    private val decoder: OutputDecoder,
    /** 模型元信息（加载时自省） */
    val modelInfo: ModelInfo,
) : FaceDetector {

    private val executor = Executors.newSingleThreadExecutor()
    private val mainHandler: Handler? = try {
        if (config.callbacksOnMainThread) Handler(Looper.getMainLooper()) else null
    } catch (t: Throwable) {
        null
    }

    @Volatile
    private var closed = false

    companion object {
        private const val TAG = "YuNetDetector"

        /**
         * 同步创建检测器（含模型加载与结构自省，约几毫秒到几十毫秒）。
         * 失败抛出 [YuNetModelException] / [YuNetIncompatibleModelException]。
         */
        @JvmStatic
        @Throws(YuNetException::class)
        fun create(context: Context, config: YuNetConfig): YuNetDetector {
            val engine = TfliteEngine(config)
            val info = engine.open(context.applicationContext)
            val pre = Preprocessor(config, info.inputChannels, info.inputLayout, info.inputWidth, info.inputHeight)
            // 解码器必须与引擎自省确定的输出索引保持一致（名称识别优先）
            val indices = engine.resolvedIndices
                ?: config.outputIndices
                ?: OutputIndices.default()
            val dec = RawYuNetDecoder(
                strides = info.strides,
                scoreThreshold = config.scoreThreshold,
                nmsThreshold = config.nmsThreshold,
                topK = config.topK,
                nmsPreTopK = config.nmsPreTopK,
                applySigmoid = config.applySigmoidToScores,
                indices = indices,
            )
            return YuNetDetector(config, engine, pre, dec, info)
        }

        /** 异步创建检测器，不阻塞调用线程 */
        @JvmStatic
        fun createAsync(context: Context, config: YuNetConfig, callback: CreateCallback) {
            Thread {
                try {
                    val d = create(context, config)
                    callback.onCreated(d)
                } catch (t: Throwable) {
                    val e = t as? YuNetException
                        ?: YuNetModelException("创建检测器失败: ${t.message}", t)
                    callback.onError(e)
                }
            }.start()
        }
    }

    // ---------------- 同步检测 ----------------

    override fun detectSync(bitmap: Bitmap): List<Face> {
        if (closed) throw YuNetClosedException("检测器已关闭")
        validateBitmap(bitmap)
        val processed = preprocessor.process(bitmap, ExifTransform.IDENTITY)
        val outputs = engine.run(processed.pixels)
        val raws = decode(outputs)
        val mapper = buildMapper(
            processed,
            finalScale = processed.inputScale,
            finalW = bitmap.width,
            finalH = bitmap.height,
        )
        return raws.map { toFace(it, mapper) }
    }

    /**
     * 从文件路径检测。自动处理：采样解码（防 OOM）、EXIF 转正、
     * 结果坐标映射回**文件全分辨率存储像素**坐标系。
     *
     * ⚠️ 坐标系说明：返回值基于文件存储像素（未施加 EXIF 变换的宽高），
     * 不是 EXIF 转正后的图像坐标。画框到转正后的 ImageView、或用于裁剪定位时，
     * 请使用 [detectFileSyncDetailed]（同时给出两组坐标）。
     */
    @Throws(YuNetException::class)
    fun detectFileSync(path: String): List<Face> = detectFileInternal(path).faces

    /**
     * 从文件路径检测，返回结构化结果与**两套坐标**：
     * [FileDetectionResult.faces]（文件存储像素坐标，与 [detectFileSync] 一致）
     * 与 [FileDetectionResult.uprightFaces]（EXIF 转正后坐标）。
     */
    @JvmOverloads
    @Throws(YuNetException::class)
    fun detectFileSyncDetailed(path: String): FileDetectionResult = detectFileInternal(path)

    private fun detectFileInternal(path: String): FileDetectionResult {
        if (closed) throw YuNetClosedException("检测器已关闭")
        val file = File(path)
        if (!file.exists() || !file.isFile) throw YuNetInputException("图片文件不存在: $path")
        if (!file.canRead()) throw YuNetInputException("图片文件不可读: $path")

        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeFile(path, bounds)
        if (bounds.outWidth <= 0 || bounds.outHeight <= 0) {
            throw YuNetInputException("无法解码图片: $path")
        }
        var sample = 1
        val maxDim = if (config.maxInputImageDim > 0) config.maxInputImageDim else Int.MAX_VALUE
        while (max(bounds.outWidth, bounds.outHeight) / (sample * 2) >= maxDim) sample *= 2
        val opts = BitmapFactory.Options().apply { inSampleSize = sample }
        val bitmap = BitmapFactory.decodeFile(path, opts)
            ?: throw YuNetInputException("图片解码失败: $path")
        try {
            val transform = if (config.autoRotateExif) exifTransform(file) else ExifTransform.IDENTITY
            val processed = preprocessor.process(bitmap, transform)
            val outputs = engine.run(processed.pixels)
            val raws = decode(outputs)
            // 竖直位图 -> 全分辨率：可能经过 采样(sample) 与 预处理下采样(inputScale) 两级
            val totalScale = sample.toFloat() * processed.inputScale
            val fileMapper = buildMapper(
                processed,
                finalScale = totalScale,
                finalW = bounds.outWidth,
                finalH = bounds.outHeight,
            )
            // 转正坐标系：跳过 EXIF 逆变换（rotation/mirror 停在竖直位图这一级）
            val uprightMapper = buildUprightMapper(
                processed,
                finalScale = totalScale,
                finalW = transform.uprightSize(bounds.outWidth, bounds.outHeight)[0],
                finalH = transform.uprightSize(bounds.outWidth, bounds.outHeight)[1],
            )
            val fileFaces = raws.map { toFace(it, fileMapper) }
            val uprightFaces = raws.map { toFace(it, uprightMapper) }
            return FileDetectionResult(
                rotationDegrees = transform.rotationDegrees,
                mirrored = transform.mirrored,
                fileWidth = bounds.outWidth,
                fileHeight = bounds.outHeight,
                faces = fileFaces,
                uprightFaces = uprightFaces,
            )
        } finally {
            runCatching { bitmap.recycle() }
        }
    }

    // ---------------- 异步检测 ----------------

    override fun detectAsync(bitmap: Bitmap, callback: DetectCallback) {
        if (closed) {
            postError(callback, YuNetClosedException("检测器已关闭"))
            return
        }
        executor.execute {
            try {
                val faces = detectSync(bitmap)
                postSuccess(callback, faces)
            } catch (t: Throwable) {
                postError(callback, wrap(t))
            }
        }
    }

    fun detectFileAsync(path: String, callback: DetectCallback) {
        if (closed) {
            postError(callback, YuNetClosedException("检测器已关闭"))
            return
        }
        executor.execute {
            try {
                val faces = detectFileSync(path)
                postSuccess(callback, faces)
            } catch (t: Throwable) {
                postError(callback, wrap(t))
            }
        }
    }

    // ---------------- 协程挂起函数（Kotlin 专用，Java 不可见） ----------------

    @JvmSynthetic
    suspend fun detect(bitmap: Bitmap): List<Face> = withContext(Dispatchers.IO) { detectSync(bitmap) }

    @JvmSynthetic
    suspend fun detectFile(path: String): List<Face> = withContext(Dispatchers.IO) { detectFileSync(path) }

    @JvmSynthetic
    suspend fun detectFileDetailed(path: String): FileDetectionResult =
        withContext(Dispatchers.IO) { detectFileSyncDetailed(path) }

    // ---------------- 生命周期 ----------------

    fun isClosed(): Boolean = closed

    override fun close() {
        if (closed) return
        closed = true
        runCatching { engine.close() }
        executor.shutdown()
    }

    // ---------------- 内部 ----------------

    private fun decode(outputs: Array<FloatArray>): List<RawFace> {
        return try {
            decoder.decode(outputs.toList(), modelInfo.inputWidth, modelInfo.inputHeight)
        } catch (e: IllegalArgumentException) {
            throw YuNetIncompatibleModelException(e.message ?: "输出解码失败")
        }
    }

    private fun buildMapper(
        processed: Preprocessor.Result,
        finalScale: Float,
        finalW: Int,
        finalH: Int,
    ): CoordinateMapper = CoordinateMapper(
        resizeMode = config.resizeMode,
        scale = processed.scale,
        scaleX = processed.scaleX,
        scaleY = processed.scaleY,
        padX = processed.padX,
        padY = processed.padY,
        rotationDeg = processed.rotationDeg,
        uprightW = processed.uprightW,
        uprightH = processed.uprightH,
        finalW = finalW,
        finalH = finalH,
        finalScale = finalScale,
        mirrored = processed.mirrored,
    )

    /**
     * 网络输入坐标 -> 竖直位图坐标的映射器（rotationDeg=0：不含 EXIF 逆变换）。
     */
    private fun buildUprightMapper(
        processed: Preprocessor.Result,
        finalScale: Float,
        finalW: Int,
        finalH: Int,
    ): CoordinateMapper = CoordinateMapper(
        resizeMode = config.resizeMode,
        scale = processed.scale,
        scaleX = processed.scaleX,
        scaleY = processed.scaleY,
        padX = processed.padX,
        padY = processed.padY,
        rotationDeg = 0f,
        uprightW = processed.uprightW,
        uprightH = processed.uprightH,
        finalW = finalW,
        finalH = finalH,
        finalScale = finalScale,
    )

    private fun toFace(raw: RawFace, mapper: CoordinateMapper): Face {
        val box = mapper.mapBox(raw.left, raw.top, raw.right, raw.bottom)
        val lms = raw.landmarks?.let { l ->
            val out = FloatArray(10)
            for (k in 0 until 5) {
                val p = mapper.mapPoint(l[k * 2], l[k * 2 + 1])
                out[k * 2] = p[0]
                out[k * 2 + 1] = p[1]
            }
            out
        }
        return Face.of(box[0], box[1], box[2], box[3], raw.score, lms)
    }

    private fun validateBitmap(bitmap: Bitmap?) {
        if (bitmap == null) throw YuNetInputException("输入 bitmap 不能为空")
        if (bitmap.isRecycled) throw YuNetInputException("输入 bitmap 已回收")
        if (bitmap.width <= 0 || bitmap.height <= 0) {
            throw YuNetInputException("输入 bitmap 尺寸非法: ${bitmap.width}x${bitmap.height}")
        }
    }

    /** 从文件读取 EXIF 方向语义（含镜像）；读不出时视为无变换 */
    private fun exifTransform(file: File): ExifTransform = runCatching {
        val exif = ExifInterface(file.absolutePath)
        ExifTransform.of(exif.rotationDegrees, exif.isFlipped)
    }.getOrDefault(ExifTransform.IDENTITY)

    private fun wrap(t: Throwable): YuNetException =
        t as? YuNetException ?: YuNetInferenceException("检测失败: ${t.message}", t)

    private fun postSuccess(callback: DetectCallback, faces: List<Face>) {
        val h = mainHandler
        if (h != null) h.post { callback.onSuccess(faces) } else callback.onSuccess(faces)
    }

    private fun postError(callback: DetectCallback, e: YuNetException) {
        val h = mainHandler
        if (h != null) h.post { callback.onError(e) } else callback.onError(e)
    }
}
