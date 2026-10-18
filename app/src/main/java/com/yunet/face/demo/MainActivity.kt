package com.yunet.face.demo

import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.view.WindowInsets
import android.view.View
import android.widget.Button
import android.widget.ScrollView
import android.widget.SeekBar
import android.widget.Spinner
import android.widget.Switch
import android.widget.TextView
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.exifinterface.media.ExifInterface
import com.yunet.face.ChannelOrder
import com.yunet.face.DelegateType
import com.yunet.face.DetectCallback
import com.yunet.face.Face
import com.yunet.face.ResizeMode
import com.yunet.face.YuNetConfig
import com.yunet.face.YuNetDetector
import com.yunet.face.YuNetDiagnostics
import com.yunet.face.YuNetException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.MainScope
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.util.concurrent.Executors
import kotlin.math.max

/**
 * YuNetFaceDetect 集成 Demo。
 *
 * 覆盖点：
 * - 模型来源：assets（yunet_fp16.tflite，随 demo 一起打包）
 * - 三种调用方式：detectAsync / detectSync / 协程 detect()
 * - 关键配置：scoreThreshold、resizeMode(LETTERBOX/STRETCH)、channelOrder(BGR/RGB)、delegate(CPU/GPU)
 * - 自检工具：YuNetDiagnostics.inspect / selfTest
 * - Java 互操作：JavaInteropDemo
 * - 人脸裁剪入口：ArchiveFaceCrop（当前图 -> 最大脸 3:4 半身照）
 * - 结果可视化：FaceOverlayView 绘制人脸框 + 5 关键点 + 双眼连线 + 倾角
 *
 * 检测坐标全部为「传入图片」的像素坐标，因此叠加显示可直接目视核对坐标回溯是否正确。
 */
class MainActivity : ComponentActivity() {

    private lateinit var overlay: FaceOverlayView
    private lateinit var tvStatus: TextView
    private lateinit var tvStats: TextView
    private lateinit var tvLog: TextView
    private lateinit var tvThreshold: TextView
    private lateinit var sbThreshold: SeekBar

    // 裁剪参数（UI 面板实时调节，见 bindCropUi）
    private lateinit var sbHeadroom: SeekBar
    private lateinit var tvHeadroom: TextView
    private lateinit var sbChinroom: SeekBar
    private lateinit var tvChinroom: TextView
    private lateinit var sbSide: SeekBar
    private lateinit var tvSide: TextView
    private var cropParams = ArchiveFaceCrop.Params()
    private lateinit var spCallMode: Spinner
    private lateinit var swStretch: Switch
    private lateinit var swRgb: Switch
    private lateinit var swGpu: Switch
    private lateinit var svLog: ScrollView

    private val io = Executors.newSingleThreadExecutor { r -> Thread(r, "yunet-demo-io") }
    private val main = Handler(Looper.getMainLooper())
    private val scope = MainScope()
    private val logBuf = StringBuilder()

    private var image: Bitmap? = null

    /** 最初从相册载入的原图；人脸裁剪始终以它为输入（与展示中的图可能不同） */
    private var sourceImage: Bitmap? = null

    @Volatile
    private var detector: YuNetDetector? = null

    /** 当前检测器对应的配置指纹；置空表示下次检测时重建 */
    @Volatile
    private var detectorKey: String? = null

    private var busy = false

    private val pickImage = registerForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
        if (uri != null) loadPickedImage(uri)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // Android 15+ (targetSdk 35+) 强制 edge-to-edge：让内容避开状态栏/导航栏
        enableEdgeToEdge()
        setContentView(R.layout.activity_main)
        applyWindowInsets(findViewById(R.id.root))

        overlay = findViewById(R.id.overlay)
        tvStatus = findViewById(R.id.tvStatus)
        tvStats = findViewById(R.id.tvStats)
        tvLog = findViewById(R.id.tvLog)
        tvThreshold = findViewById(R.id.tvThreshold)
        sbThreshold = findViewById(R.id.sbThreshold)
        spCallMode = findViewById(R.id.spCallMode)
        swStretch = findViewById(R.id.swStretch)
        swRgb = findViewById(R.id.swRgb)
        swGpu = findViewById(R.id.swGpu)
        svLog = findViewById(R.id.svLog)

        findViewById<Button>(R.id.btnPick).setOnClickListener {
            pickImage.launch(
                PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)
            )
        }
        findViewById<Button>(R.id.btnCrop).setOnClickListener { runFaceCrop() }
        findViewById<Button>(R.id.btnSelfTest).setOnClickListener { runSelfTest() }
        findViewById<Button>(R.id.btnJava).setOnClickListener { runJavaInterop() }
        findViewById<Button>(R.id.btnCopyLog).setOnClickListener { copyLog() }
        findViewById<Button>(R.id.btnClearLog).setOnClickListener { clearLog() }

        sbThreshold.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(bar: SeekBar?, progress: Int, fromUser: Boolean) {
                tvThreshold.text = "%.2f".format(thresholdOf(progress))
                if (fromUser) invalidateDetector()
            }

            override fun onStartTrackingTouch(bar: SeekBar?) = Unit
            override fun onStopTrackingTouch(bar: SeekBar?) = Unit
        })
        tvThreshold.text = "%.2f".format(thresholdOf(sbThreshold.progress))

        bindCropUi()
        findViewById<Button>(R.id.btnResetParams).setOnClickListener {
            // setProgress 会触发监听器，同步更新数值文案与 cropParams
            applyCropPreset(headStep = 18, chinStep = 32, sideStep = 14)
        }
        findViewById<Button>(R.id.btnPresetTight).setOnClickListener { applyCropPreset(headStep = 12, chinStep = 22, sideStep = 9) }
        findViewById<Button>(R.id.btnPresetStd).setOnClickListener { applyCropPreset(headStep = 18, chinStep = 32, sideStep = 14) }
        findViewById<Button>(R.id.btnPresetLoose).setOnClickListener { applyCropPreset(headStep = 24, chinStep = 44, sideStep = 20) }

        // 调用方式只影响 demo 的发起方，不影响检测器实例
        val configChanged = View.OnClickListener { invalidateDetector() }
        swStretch.setOnClickListener(configChanged)
        swRgb.setOnClickListener(configChanged)
        swGpu.setOnClickListener(configChanged)

        appendLog("YuNetFaceDetect demo 启动")
        appendLog("模型: assets/$MODEL_ASSET (sha256 见 YuNetDiagnostics 报告)")
    }

    override fun onDestroy() {
        super.onDestroy()
        runCatching { detector?.close() }
        detector = null
        io.shutdown()
        scope.cancel()
    }

    // ---------------------------------- 图片 ----------------------------------

    private fun loadPickedImage(uri: Uri) {
        io.execute {
            val bmp = runCatching { decodeUri(uri, MAX_DIM) }.getOrNull()
            post {
                if (bmp == null) {
                    appendLog("✗ 解码所选图片失败: $uri")
                } else {
                    // 记住最初载入的原图：裁剪永远以它为输入（而不是上一轮的裁剪结果）
                    sourceImage = bmp
                    setImage(bmp, "相册图片 (已按 EXIF 转正, ${bmp.width}x${bmp.height})")
                    startDetect()
                }
            }
        }
    }

    /** 采样解码 + EXIF 转正，与库内 detectFileSync 的处理方式保持一致 */
    private fun decodeUri(uri: Uri, maxDim: Int): Bitmap {
        val rotation = contentResolver.openInputStream(uri)?.use { raw ->
            raw.buffered().let { runCatching { ExifInterface(it).rotationDegrees }.getOrDefault(0) }
        } ?: 0

        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        contentResolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it, null, bounds) }
        check(bounds.outWidth > 0 && bounds.outHeight > 0) { "无法解码图片: $uri" }

        var sample = 1
        while (max(bounds.outWidth, bounds.outHeight) / (sample * 2) >= maxDim) sample *= 2
        val opts = BitmapFactory.Options().apply { inSampleSize = sample }
        val raw = contentResolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it, null, opts) }
            ?: error("图片解码失败: $uri")

        if (rotation == 0) return raw
        val rotated = Bitmap.createBitmap(
            raw, 0, 0, raw.width, raw.height,
            Matrix().apply { postRotate(rotation.toFloat()) }, true
        )
        if (rotated !== raw) raw.recycle()
        return rotated
    }

    private fun setImage(bmp: Bitmap, desc: String) {
        image = bmp
        overlay.setImage(bmp)
        tvStats.text = "图片 ${bmp.width}x${bmp.height}"
        appendLog("· 载入图片 $desc")
    }

    // ---------------------------------- 人脸裁剪 ----------------------------------

    /**
     * 人脸裁剪入口（ArchiveFaceCrop）：把当前展示图裁成以最大脸为中心的 3:4 半身照。
     * Photo Picker 的 Uri 不是文件路径，先把展示位图落到 cacheDir，再以文件路径交给裁剪器。
     */
    private fun runFaceCrop() {
        // 裁剪始终对原图进行：调参反复点「人脸裁剪」也是基于同一份原图，不会拿上一轮结果继续裁
        val bmp = sourceImage ?: image
        if (bmp == null) {
            toast(getString(R.string.hint_pick_image))
            return
        }
        if (busy) return
        busy = true
        tvStats.text = getString(R.string.hint_busy)
        appendLog("· 人脸裁剪（ArchiveFaceCrop：始终以原图为输入，最大脸水平校正 + 外扩）")
        appendLog(
            "   · 参数: 头顶=%.2f 下方=%.2f 左右=%.2f 比例=%.2f".format(
                cropParams.headroomScale, cropParams.chinroomScale, cropParams.sideScale, cropParams.aspect,
            ),
        )
        scope.launch {
            try {
                val src = withContext(Dispatchers.IO) { writeTempJpeg(bmp) }
                val out = ArchiveFaceCrop.crop(this@MainActivity, src, cropParams) { msg -> appendLog("   · $msg") }
                if (out == null) {
                    post {
                        busy = false
                        appendLog("✗ 裁剪放弃：未检出人脸或处理异常，保留原图")
                    }
                } else {
                    val cropped = withContext(Dispatchers.IO) { decodeFile(out, MAX_DIM) }
                    post {
                        if (cropped == null) {
                            busy = false
                            appendLog("✗ 裁剪结果解码失败: $out")
                        } else {
                            appendLog("✓ 裁剪完成: $out")
                            busy = false
                            setImage(cropped, "人脸裁剪结果 (${cropped.width}x${cropped.height})")
                            startDetect()
                        }
                    }
                }
            } catch (t: Throwable) {
                fail(t)
            }
        }
    }

    private fun writeTempJpeg(bmp: Bitmap): String {
        val f = File(cacheDir, "crop_input.jpg")
        f.parentFile?.mkdirs()
        f.outputStream().use { bmp.compress(Bitmap.CompressFormat.JPEG, 95, it) }
        return f.absolutePath
    }

    /** 采样解码本地文件（防 OOM），失败返回 null。 */
    private fun decodeFile(path: String, maxDim: Int): Bitmap? = runCatching {
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeFile(path, bounds)
        if (bounds.outWidth <= 0 || bounds.outHeight <= 0) return null
        var s = 1
        while (max(bounds.outWidth, bounds.outHeight) / (s * 2) >= maxDim) s *= 2
        BitmapFactory.decodeFile(path, BitmapFactory.Options().apply { inSampleSize = s })
    }.getOrNull()

    // ---------------------------------- 检测 ----------------------------------

    private fun startDetect() {
        val bmp = image
        if (bmp == null) {
            toast(getString(R.string.hint_pick_image))
            return
        }
        if (busy) return
        busy = true
        tvStats.text = getString(R.string.hint_busy)
        overlay.setFaces(emptyList())

        val config = buildConfig()          // 读取控件：必须在主线程
        val mode = CallMode.entries[spCallMode.selectedItemPosition]

        io.execute {
            val d = try {
                ensureDetector(config)
            } catch (t: Throwable) {
                fail(t)
                return@execute
            }
            when (mode) {
                CallMode.SYNC -> {
                    val t0 = System.currentTimeMillis()
                    try {
                        val faces = d.detectSync(bmp)
                        post { onFaces(faces, System.currentTimeMillis() - t0, "detectSync(后台线程)") }
                    } catch (t: Throwable) {
                        fail(t)
                    }
                }

                CallMode.ASYNC -> post {
                    val t0 = System.currentTimeMillis()
                    d.detectAsync(bmp, object : DetectCallback {
                        override fun onSuccess(faces: List<Face>) {
                            onFaces(faces, System.currentTimeMillis() - t0, "detectAsync(回调)")
                        }

                        override fun onError(error: YuNetException) = fail(error)
                    })
                }

                CallMode.COROUTINE -> post {
                    val t0 = System.currentTimeMillis()
                    scope.launch {
                        try {
                            onFaces(d.detect(bmp), System.currentTimeMillis() - t0, "detect(挂起函数)")
                        } catch (t: Throwable) {
                            fail(t)
                        }
                    }
                }
            }
        }
    }

    /** 仅可在 io 线程调用：配置指纹变化时重建检测器 */
    private fun ensureDetector(config: YuNetConfig): YuNetDetector {
        val key = configKey(config)
        val existing = detector
        if (existing != null && !existing.isClosed() && key == detectorKey) return existing

        existing?.let { runCatching { it.close() } }
        val t0 = System.currentTimeMillis()
        val d = YuNetDetector.create(this, config)
        val ms = System.currentTimeMillis() - t0
        detector = d
        detectorKey = key

        val info = d.modelInfo
        // ASSET 来源时 ModelInfo 无法拿到文件长度（-1），这里从 AssetInputStream 取
        val sizeKb = if (info.modelSizeBytes > 0) {
            info.modelSizeBytes / 1024
        } else {
            runCatching { assets.open(MODEL_ASSET).use { it.available().toLong() / 1024 } }.getOrDefault(0L)
        }
        post {
            tvStatus.text = buildString {
                append("模型 assets/$MODEL_ASSET ${sizeKb}KB ")
                append("输入 ${info.inputShape.toList()} ${info.inputLayout}/${info.inputDtype}\n")
                append("输出 ${info.outputCount} 路 strides=${info.strides.toList()} ")
                append("anchors=${info.totalAnchors} 名称映射=${info.mappingFromNames}\n")
                append("配置 $key")
            }
            appendLog("—— 创建检测器: $key, 加载耗时 ${ms}ms")
            appendLog(info.toString())
        }
        return d
    }

    private fun onFaces(faces: List<Face>, ms: Long, api: String) {
        busy = false
        overlay.setFaces(faces)
        val bmp = image
        tvStats.text = "$api ${ms}ms / 检出 ${faces.size} (图片 ${bmp?.width}x${bmp?.height})"
        appendLog("· $api 耗时 ${ms}ms, 检出 ${faces.size} 张人脸")
        faces.forEachIndexed { i, f ->
            appendLog(
                "   #%d score=%.3f box=[%.0f,%.0f,%.0f,%.0f] 尺寸=%.0fx%.0f 倾角=%+.1f°".format(
                    i, f.score, f.box.left, f.box.top, f.box.right, f.box.bottom,
                    f.width, f.height, f.eyeAngleDegrees()
                )
            )
            val le = f.leftEye
            val re = f.rightEye
            val nose = f.nose
            if (le != null && re != null && nose != null) {
                appendLog(
                    "     关键点 眼=(%.0f,%.0f)-(%.0f,%.0f) 鼻=(%.0f,%.0f)".format(
                        le.x, le.y, re.x, re.y, nose.x, nose.y
                    )
                )
            }
        }
        if (faces.isEmpty()) {
            appendLog("   0 检出：可把阈值降到 0.30，或切换「拉伸 / RGB」开关后重试")
        }
    }

    private fun fail(t: Throwable) {
        post {
            busy = false
            val name = t.javaClass.simpleName
            appendLog("✗ $name: ${t.message}")
            Log.w(TAG, "检测失败", t)
            tvStats.text = "失败: $name"
            if (t !is YuNetException) appendLog(t.stackTraceToString())
        }
    }

    // ---------------------------------- 自检 / Java 互操作 ----------------------------------

    private fun runSelfTest() {
        val config = buildConfig()
        val bmp = image
        appendLog("· 开始自检（inspect + selfTest${if (bmp == null) "，无图片用内置灰度图" else ""}）")
        io.execute {
            val out = StringBuilder()
            try {
                out.append(YuNetDiagnostics.inspect(this, config)).append('\n')
                out.append(YuNetDiagnostics.selfTest(this, config, bmp))
            } catch (t: Throwable) {
                out.append("自检失败: ").append(t.javaClass.simpleName).append(": ").append(t.message)
            }
            post { appendLog(out.toString().trimEnd()) }
        }
    }

    private fun runJavaInterop() {
        val bmp = image
        if (bmp == null) {
            toast(getString(R.string.hint_pick_image))
            return
        }
        appendLog("· Java 互操作调用（JavaInteropDemo.detect）")
        JavaInteropDemo.detect(this, bmp) { summary, count ->
            appendLog(summary)
            if (count >= 0) tvStats.text = "Java 互操作: 检出 $count"
        }
    }

    // ---------------------------------- 配置 ----------------------------------

    private fun thresholdOf(progress: Int): Float = (0.05f + progress * 0.01f).coerceIn(0f, 1f)

    // ---- 裁剪参数面板（档位见 applyCropPreset；滑杆步长 0.05×人脸框倍数） ----

    private fun cropScaleOf(progress: Int): Float = progress * 0.05f

    /** 面板滑杆 ↔ ArchiveFaceCrop.Params 双向联动：拖动即改，点「人脸裁剪」时生效。 */
    private fun bindCropUi() {
        sbHeadroom = findViewById(R.id.sbHeadroom)
        tvHeadroom = findViewById(R.id.tvHeadroom)
        sbChinroom = findViewById(R.id.sbChinroom)
        tvChinroom = findViewById(R.id.tvChinroom)
        sbSide = findViewById(R.id.sbSide)
        tvSide = findViewById(R.id.tvSide)

        fun bind(bar: SeekBar, tv: TextView, apply: (Float) -> Unit) {
            bar.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
                override fun onProgressChanged(b: SeekBar?, progress: Int, fromUser: Boolean) {
                    tv.text = "%.2f".format(cropScaleOf(progress))
                    apply(cropScaleOf(progress))
                }

                override fun onStartTrackingTouch(b: SeekBar?) = Unit
                override fun onStopTrackingTouch(b: SeekBar?) = Unit
            })
            tv.text = "%.2f".format(cropScaleOf(bar.progress))
        }
        bind(sbHeadroom, tvHeadroom) { cropParams = cropParams.copy(headroomScale = it) }
        bind(sbChinroom, tvChinroom) { cropParams = cropParams.copy(chinroomScale = it) }
        bind(sbSide, tvSide) { cropParams = cropParams.copy(sideScale = it) }
    }

    /** 三档/复位统一入口：走 SeekBar.progress 赋值，经同一套监听器刷新文案与参数。 */
    private fun applyCropPreset(headStep: Int, chinStep: Int, sideStep: Int) {
        sbHeadroom.progress = headStep
        sbChinroom.progress = chinStep
        sbSide.progress = sideStep
    }

    private fun buildConfig(): YuNetConfig = YuNetConfig.Builder()
        .useAsset(MODEL_ASSET)
        .scoreThreshold(thresholdOf(sbThreshold.progress))
        .nmsThreshold(0.45f)
        .topK(100)
        .nmsPreTopK(5000)
        .resizeMode(if (swStretch.isChecked) ResizeMode.STRETCH else ResizeMode.LETTERBOX)
        .channelOrder(if (swRgb.isChecked) ChannelOrder.RGB else ChannelOrder.BGR)
        .delegate(if (swGpu.isChecked) DelegateType.GPU else DelegateType.CPU)
        .maxInputImageDim(2048)
        .callbacksOnMainThread(true)
        .build()

    private fun configKey(c: YuNetConfig): String = "thr=%.2f,%s,%s,%s".format(
        c.scoreThreshold, c.resizeMode, c.channelOrder, c.delegate
    )

    private fun invalidateDetector() {
        detectorKey = null
    }

    // ---------------------------------- 日志 ----------------------------------

    private fun appendLog(text: String) {
        if (Looper.myLooper() != Looper.getMainLooper()) {
            post { appendLog(text) }
            return
        }
        logBuf.append(text).append('\n')
        if (logBuf.length > 100_000) logBuf.delete(0, logBuf.length - 100_000)
        tvLog.text = logBuf.toString()
        svLog.post { svLog.fullScroll(View.FOCUS_DOWN) }
        // 同步输出到 logcat，便于 adb logcat -s YuNetDemo 观察
        Log.i(TAG, text.lineSequence().joinToString(" | "))
    }

    private fun clearLog() {
        logBuf.setLength(0)
        tvLog.text = ""
    }

    private fun copyLog() {
        if (logBuf.isEmpty()) {
            toast(getString(R.string.toast_empty))
            return
        }
        (getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager)
            .setPrimaryClip(ClipData.newPlainText("YuNet Demo Log", logBuf.toString()))
        toast(getString(R.string.toast_copied))
    }

    // ---------------------------------- 工具 ----------------------------------

    private fun post(block: () -> Unit) {
        if (Looper.myLooper() == Looper.getMainLooper()) block() else main.post(block)
    }

    private fun toast(msg: String) {
        Toast.makeText(this, msg, Toast.LENGTH_SHORT).show()
    }

    /** edge-to-edge 下把系统栏高度加进根布局 padding，避免内容被状态栏/导航栏遮挡 */
    @Suppress("DEPRECATION")
    private fun applyWindowInsets(root: View) {
        val base = (8 * resources.displayMetrics.density).toInt()
        root.setOnApplyWindowInsetsListener { v, insets ->
            val top: Int
            val bottom: Int
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
                val bars = insets.getInsets(
                    WindowInsets.Type.systemBars() or WindowInsets.Type.displayCutout()
                )
                top = bars.top
                bottom = bars.bottom
            } else {
                top = insets.systemWindowInsetTop
                bottom = insets.systemWindowInsetBottom
            }
            v.setPadding(base, base + top, base, base + bottom)
            insets
        }
    }

    private enum class CallMode { ASYNC, SYNC, COROUTINE }

    private companion object {
        const val TAG = "YuNetDemo"
        const val MODEL_ASSET = "yunet_fp16.tflite"
        const val MAX_DIM = 2048
    }
}
