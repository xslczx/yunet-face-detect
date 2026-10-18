package com.yunet.face.demo

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.os.Looper
import android.util.Log
import androidx.exifinterface.media.ExifInterface
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.yunet.face.ChannelOrder
import com.yunet.face.CreateCallback
import com.yunet.face.DetectCallback
import com.yunet.face.Face
import com.yunet.face.ResizeMode
import com.yunet.face.YuNetClosedException
import com.yunet.face.YuNetConfig
import com.yunet.face.YuNetDetector
import com.yunet.face.YuNetDiagnostics
import com.yunet.face.YuNetException
import kotlinx.coroutines.runBlocking
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.io.FileOutputStream
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicReference
import kotlin.math.abs

/**
 * 设备端（真机 / 模拟器）集成测试：跑通「模型加载 -> 预处理 -> 推理 -> 解码 -> 坐标回溯」全链路。
 *
 * 运行： ./gradlew :app:connectedDebugAndroidTest
 *
 * 与 `:yunetfacedetect:testDebugUnitTest`（纯 JVM，覆盖解码/NMS/坐标映射公式）互补，
 * 这里验证的是依赖 Android 框架和 TFLite native 的部分。
 */
@RunWith(AndroidJUnit4::class)
class YuNetDeviceTest {

    private lateinit var ctx: Context
    private lateinit var sample: Bitmap
    private val detectors = mutableListOf<YuNetDetector>()

    @Before
    fun setUp() {
        ctx = InstrumentationRegistry.getInstrumentation().targetContext
        // 示例图只随 androidTest APK 打包（app/src/androidTest/assets/），需从 instrumentation 上下文读取
        val bmp = InstrumentationRegistry.getInstrumentation().context.assets.open(SAMPLE)
            .use { BitmapFactory.decodeStream(it) }
        assertNotNull("androidTest assets/$SAMPLE 解码失败", bmp)
        sample = bmp!!
        Log.i(TAG, "示例图片 ${sample.width}x${sample.height}")
    }

    @After
    fun tearDown() {
        detectors.forEach { runCatching { it.close() } }
        detectors.clear()
        if (::sample.isInitialized) sample.recycle()
    }

    // ---------------------------- 结构自检 ----------------------------

    @Test
    fun modelStructureIsCompatible() {
        val report = YuNetDiagnostics.inspect(ctx, config())
        Log.i(TAG, report)
        assertTrue("输出应为 12 路", report.contains("outputs=12"))
        assertTrue("anchor 总数应为 8400", report.contains("total=8400"))
        assertTrue("输入应为 NCHW", report.contains("NCHW"))
        assertTrue("I/O 应为 FLOAT32", report.contains("FLOAT32"))
        assertTrue("应识别出 8/16/32 三个 stride", report.contains("[8, 16, 32]"))
        // 实测 litert-community 转换版的输出名为 serving_default_output_N_output，
        // 不含 cls/obj/bbox/kps 关键字，因此走「默认索引顺序」而非名称映射（可接受，见 README 1.5）
        Log.i(TAG, "mappingFromNames=${report.contains("mappingFromNames=true")}")
        assertTrue(
            "输出角色应按默认顺序标注为 cls/obj/bbox/kps x stride",
            report.contains("cls(stride=8)") && report.contains("obj(stride=16)") &&
                report.contains("bbox(stride=32)") && report.contains("kps(stride=8)")
        )
    }

    // ---------------------------- 同步 / 异步 / 协程 ----------------------------

    @Test
    fun detectSyncFindsFacesOnSamplePhoto() {
        val d = detector()
        val t0 = System.currentTimeMillis()
        val faces = d.detectSync(sample)
        Log.i(TAG, "detectSync 耗时 ${System.currentTimeMillis() - t0}ms, 检出 ${faces.size}")
        faces.take(5).forEach { Log.i(TAG, "  $it") }

        assertTrue("示例合照应至少检出 1 张人脸", faces.isNotEmpty())
        assertTrue("largest_selfie 为多人合照, 预期 >= 3 张, 实际 ${faces.size}", faces.size >= 3)

        for (f in faces) {
            assertTrue("score 应在 (0,1]: ${f.score}", f.score > 0f && f.score <= 1.0001f)
            assertTrue("框宽高应为正: ${f.width}x${f.height}", f.width > 0f && f.height > 0f)
            assertTrue(
                "框中心应落在图片范围内: ${f.box.centerX()},${f.box.centerY()}",
                f.box.centerX() in 0f..sample.width.toFloat() &&
                    f.box.centerY() in 0f..sample.height.toFloat()
            )
            val lm = f.landmarks
            assertNotNull("关键点应解码成功", lm)
            assertEquals("关键点应为 5 点 x/y 共 10 个 float", 10, lm!!.size)
            assertTrue("关键点应在框附近", lm.all { it.isFinite() })
            val angle = f.eyeAngleDegrees()
            assertTrue("正脸倾角应在 ±45° 内, 实际 $angle", abs(angle) < 45f)
        }

        // 结果按分数降序
        for (i in 1 until faces.size) {
            assertTrue("结果应按分数降序", faces[i - 1].score >= faces[i].score)
        }
    }

    @Test
    fun detectAsyncCallbackDeliveredOnMainThread() {
        val d = detector()
        val resultFaces = AtomicReference<List<Face>>()
        val resultError = AtomicReference<YuNetException>()
        val onMainThread = AtomicReference(false)
        val latch = CountDownLatch(1)

        d.detectAsync(sample, object : DetectCallback {
            override fun onSuccess(faces: List<Face>) {
                onMainThread.set(Looper.myLooper() == Looper.getMainLooper())
                resultFaces.set(faces)
                latch.countDown()
            }

            override fun onError(error: YuNetException) {
                resultError.set(error)
                latch.countDown()
            }
        })

        assertTrue("异步回调超时", latch.await(TIMEOUT_S, TimeUnit.SECONDS))
        assertNull("异步不应报错: ${resultError.get()?.message}", resultError.get())
        assertTrue("callbacksOnMainThread=true 时回调应在主线程", onMainThread.get())
        assertTrue("异步结果应与同步一致（非空）", resultFaces.get()?.isNotEmpty() == true)
    }

    @Test
    fun detectSuspendFunctionWorks() = runBlocking {
        val d = detector()
        val t0 = System.currentTimeMillis()
        val faces = d.detect(sample)
        Log.i(TAG, "detect(挂起) 耗时 ${System.currentTimeMillis() - t0}ms, 检出 ${faces.size}")
        assertTrue(faces.isNotEmpty())
    }

    @Test
    fun createAsyncWorks() {
        val holder = AtomicReference<YuNetDetector>()
        val createError = AtomicReference<YuNetException>()
        val latch = CountDownLatch(1)

        YuNetDetector.createAsync(ctx, config(), object : CreateCallback {
            override fun onCreated(detector: YuNetDetector) {
                holder.set(detector)
                latch.countDown()
            }

            override fun onError(error: YuNetException) {
                createError.set(error)
                latch.countDown()
            }
        })

        assertTrue("createAsync 超时", latch.await(TIMEOUT_S, TimeUnit.SECONDS))
        assertNull("createAsync 不应报错: ${createError.get()?.message}", createError.get())
        val d = holder.get() ?: throw AssertionError("createAsync 未返回检测器")
        detectors += d
        assertTrue(d.detectSync(sample).isNotEmpty())
    }

    // ---------------------------- 配置项 ----------------------------

    @Test
    fun letterboxAndStretchBothDetect() {
        val lb = detector(config { resizeMode(ResizeMode.LETTERBOX) }).detectSync(sample)
        val st = detector(config { resizeMode(ResizeMode.STRETCH) }).detectSync(sample)
        Log.i(TAG, "LETTERBOX=${lb.size}, STRETCH=${st.size}")
        assertTrue("LETTERBOX 应有检出", lb.isNotEmpty())
        assertTrue("STRETCH 应有检出", st.isNotEmpty())
    }

    @Test
    fun channelOrderBgrIsDefaultAndRgbStillWorks() {
        val bgr = detector(config { channelOrder(ChannelOrder.BGR) }).detectSync(sample)
        val rgb = detector(config { channelOrder(ChannelOrder.RGB) }).detectSync(sample)
        Log.i(TAG, "BGR=${bgr.size}, RGB=${rgb.size}")
        assertTrue("BGR（官方顺序）应有检出", bgr.isNotEmpty())
        assertTrue("BGR 检出数应不少于 RGB", bgr.size >= rgb.size)
    }

    @Test
    fun scoreThresholdAndTopKAreHonored() {
        val loose = detector(config { scoreThreshold(0.3f) }).detectSync(sample)
        val strict = detector(config { scoreThreshold(0.9f) }).detectSync(sample)
        Log.i(TAG, "thr0.3=${loose.size}, thr0.9=${strict.size}")
        assertTrue("阈值 0.3 应有检出", loose.isNotEmpty())
        assertTrue("高阈值检出数不应更多", strict.size <= loose.size)

        val top2 = detector(config { topK(2) }).detectSync(sample)
        assertTrue("topK=2 应最多返回 2 张, 实际 ${top2.size}", top2.size <= 2)
    }

    // ---------------------------- 文件 / EXIF ----------------------------

    @Test
    fun detectFileSyncHandlesLargeImageAndExifRotation() {
        val d = detector()

        // 1) 直接落盘（无 EXIF 旋转）
        val plain = File(ctx.cacheDir, "yunet-test-plain.jpg")
        FileOutputStream(plain).use { sample.compress(Bitmap.CompressFormat.JPEG, 95, it) }
        val fromPlain = d.detectFileSync(plain.absolutePath)
        Log.i(TAG, "detectFileSync(plain) 检出 ${fromPlain.size}")
        assertTrue("文件路径检测应有检出", fromPlain.isNotEmpty())

        // 2) 真实世界的 EXIF orientation=6 文件：像素按 -90° 存储 + 标签「需顺时针 90° 转正」
        //    这样读取方 +90° 顺时针转正后得到的才是原图（若像素本就正、只贴标签，是语义错误的文件）
        val stored = Bitmap.createBitmap(
            sample, 0, 0, sample.width, sample.height,
            Matrix().apply { postRotate(-90f) }, true
        )
        val rotated = File(ctx.cacheDir, "yunet-test-exif6.jpg")
        FileOutputStream(rotated).use { stored.compress(Bitmap.CompressFormat.JPEG, 95, it) }
        ExifInterface(rotated.absolutePath).apply {
            setAttribute(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_ROTATE_90.toString())
            saveAttributes()
        }
        assertEquals("EXIF 标签应被写入", 90, ExifInterface(rotated).rotationDegrees)

        val fromRotated = d.detectFileSync(rotated.absolutePath)
        Log.i(TAG, "detectFileSync(exif6) 存储尺寸=${stored.width}x${stored.height} 检出 ${fromRotated.size}")

        // 库内已按 EXIF 转正后再送网络，因此检出量应与原图相当
        assertTrue("EXIF 转正后应有检出", fromRotated.isNotEmpty())
        assertTrue(
            "EXIF 转正后的检出量应接近原图: plain=${fromPlain.size}, exif6=${fromRotated.size}",
            fromRotated.size >= fromPlain.size / 2
        )
        // 坐标映射回「文件存储像素」坐标系（CoordinateMapper 会把网络坐标逆变换回存储朝向）
        for (f in fromRotated) {
            assertTrue(
                "EXIF 逆旋转后的坐标应落在文件存储尺寸内: ${f.box.centerX()},${f.box.centerY()}",
                f.box.centerX() in 0f..stored.width.toFloat() &&
                    f.box.centerY() in 0f..stored.height.toFloat()
            )
        }
        // 3) 对照：同样的像素但不带 EXIF 标签 -> 侧躺人脸，YuNet 检不出（说明上面的检出确实来自 EXIF 转正）
        val noTag = File(ctx.cacheDir, "yunet-test-notag.jpg")
        FileOutputStream(noTag).use { stored.compress(Bitmap.CompressFormat.JPEG, 95, it) }
        val fromNoTag = d.detectFileSync(noTag.absolutePath)
        Log.i(TAG, "对照（-90° 像素、无 EXIF 标签）检出 ${fromNoTag.size}")
        assertTrue("侧躺人脸不应被检出（YuNet 只认近似正脸）", fromNoTag.size < fromRotated.size)

        stored.recycle()
        listOf(plain, rotated, noTag).forEach { it.delete() }
    }

    @Test
    fun detectFileSyncDetailedReportsBothCoordinateSystems() {
        val d = detector()
        // orientation=6 文件：存储像素为原图 -90° 侧躺 + 标签「顺时针 90° 转正」
        val stored = Bitmap.createBitmap(
            sample, 0, 0, sample.width, sample.height,
            Matrix().apply { postRotate(-90f) }, true
        )
        val rotated = File(ctx.cacheDir, "yunet-test-detailed-exif6.jpg")
        FileOutputStream(rotated).use { stored.compress(Bitmap.CompressFormat.JPEG, 95, it) }
        ExifInterface(rotated.absolutePath).apply {
            setAttribute(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_ROTATE_90.toString())
            saveAttributes()
        }
        try {
            val result = d.detectFileSyncDetailed(rotated.absolutePath)
            assertEquals(90, result.rotationDegrees)
            assertFalse("orientation 6 不含镜像", result.mirrored)
            assertEquals("fileWidth 应为存储像素宽", stored.width, result.fileWidth)
            assertEquals("fileHeight 应为存储像素高", stored.height, result.fileHeight)
            assertEquals("uprightWidth 应宽高互换", stored.height, result.uprightWidth)
            assertEquals("uprightHeight 应宽高互换", stored.width, result.uprightHeight)
            assertTrue("转正坐标应有检出", result.uprightFaces.isNotEmpty())
            assertEquals("两套坐标应与检出按索引对应", result.faces.size, result.uprightFaces.size)
            for (f in result.faces) {
                assertTrue(
                    "file 坐标应落在存储尺寸内: ${f.box.centerX()},${f.box.centerY()}",
                    f.box.centerX() in 0f..stored.width.toFloat() && f.box.centerY() in 0f..stored.height.toFloat(),
                )
            }
            for (f in result.uprightFaces) {
                assertTrue(
                    "upright 坐标应落在转正尺寸内: ${f.box.centerX()},${f.box.centerY()}",
                    f.box.centerX() in 0f..result.uprightWidth.toFloat() &&
                        f.box.centerY() in 0f..result.uprightHeight.toFloat(),
                )
            }
        } finally {
            stored.recycle()
            rotated.delete()
        }
    }

    @Test
    fun detectFileSyncMirroredExifRectifiesOrientation() {
        val d = detector()
        // 对照：原图无 EXIF 标签
        val plain = File(ctx.cacheDir, "yunet-test-mirror-plain.jpg")
        FileOutputStream(plain).use { sample.compress(Bitmap.CompressFormat.JPEG, 95, it) }
        val fromPlain = d.detectFileSync(plain.absolutePath)
        assertTrue("原图应有检出", fromPlain.isNotEmpty())

        // orientation 2：存储像素 = 原图水平镜像 + 标签「展示时水平翻转」
        // （真实流：前摄/部分修图软件导出常用该组合；不处理时按存储像素检测会框到镜像位置）
        val mirrored = Bitmap.createBitmap(
            sample, 0, 0, sample.width, sample.height,
            Matrix().apply {
                postScale(-1f, 1f)
                postTranslate(sample.width.toFloat(), 0f)
            }, true
        )
        val flipped = File(ctx.cacheDir, "yunet-test-mirror-exif2.jpg")
        FileOutputStream(flipped).use { mirrored.compress(Bitmap.CompressFormat.JPEG, 95, it) }
        ExifInterface(flipped.absolutePath).apply {
            setAttribute(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_FLIP_HORIZONTAL.toString())
            saveAttributes()
        }
        assertEquals("镜像标签应可读出", true, ExifInterface(flipped.absolutePath).isFlipped)
        try {
            val result = d.detectFileSyncDetailed(flipped.absolutePath)
            assertEquals(0, result.rotationDegrees)
            assertTrue("orientation 2 应标记镜像", result.mirrored)
            Log.i(TAG, "mirror exif2 检出 ${result.faces.size}, upright ${result.uprightFaces.size}")

            // 1) 转正像素 == 原图 → uprightFaces 顶部结果应与原图检出位置一致
            assertTrue("镜像转正后应有检出", result.uprightFaces.isNotEmpty())
            val plainTop = fromPlain[0]
            val upTop = result.uprightFaces[0]
            val tolX = sample.width * 0.02f
            val tolY = sample.height * 0.02f
            assertTrue(
                "转正坐标应与原图检出接近: upright=${upTop.box.left},${upTop.box.top} " +
                    "plain=${plainTop.box.left},${plainTop.box.top} tol=$tolX,$tolY",
                abs(upTop.box.centerX() - plainTop.box.centerX()) < tolX &&
                    abs(upTop.box.centerY() - plainTop.box.centerY()) < tolY,
            )

            // 2) file（存储像素）坐标应为原图的镜像：centerX ≈ W − plain.centerX
            val fileTop = result.faces[0]
            assertTrue(
                "存储像素坐标应为原图镜像: file=${fileTop.box.centerX()} " +
                    "期望=${sample.width - plainTop.box.centerX()}",
                abs(fileTop.box.centerX() - (sample.width - plainTop.box.centerX())) < tolX &&
                    abs(fileTop.box.centerY() - plainTop.box.centerY()) < tolY,
            )
        } finally {
            mirrored.recycle()
            plain.delete()
            flipped.delete()
        }
    }

    @Test
    fun invalidInputIsRejected() {
        val d = detector()
        val recycled = Bitmap.createBitmap(64, 64, Bitmap.Config.ARGB_8888).apply { recycle() }
        try {
            d.detectSync(recycled)
            fail("已回收的 bitmap 应抛 YuNetException")
        } catch (e: YuNetException) {
            Log.i(TAG, "已回收 bitmap 正确抛出 ${e.javaClass.simpleName}: ${e.message}")
        }
        try {
            d.detectFileSync(File(ctx.cacheDir, "not-exist.jpg").absolutePath)
            fail("不存在的文件应抛 YuNetException")
        } catch (e: YuNetException) {
            Log.i(TAG, "缺失文件正确抛出 ${e.javaClass.simpleName}: ${e.message}")
        }
    }

    @Test
    fun grayImageReturnsEmptyListWithoutCrash() {
        val d = detector()
        val gray = Bitmap.createBitmap(640, 480, Bitmap.Config.ARGB_8888).apply { eraseColor(0xFF808080.toInt()) }
        val faces = d.detectSync(gray)
        gray.recycle()
        assertEquals("纯灰图不应检出人脸", 0, faces.size)
    }

    // ---------------------------- 生命周期 ----------------------------

    @Test
    fun closingDetectorRejectsFurtherUse() {
        val d = detector()
        assertTrue(d.detectSync(sample).isNotEmpty())
        d.close()
        assertTrue("close() 后 isClosed 应为 true", d.isClosed())
        d.close() // 幂等

        try {
            d.detectSync(sample)
            fail("close() 后 detectSync 应抛 YuNetClosedException")
        } catch (e: YuNetClosedException) {
            Log.i(TAG, "close() 后正确抛出 ${e.message}")
        }

        // 异步路径不抛异常，经 onError 返回
        val closedError = AtomicReference<YuNetException>()
        val latch = CountDownLatch(1)
        d.detectAsync(sample, object : DetectCallback {
            override fun onSuccess(faces: List<Face>) = latch.countDown()
            override fun onError(error: YuNetException) {
                closedError.set(error)
                latch.countDown()
            }
        })
        assertTrue(latch.await(TIMEOUT_S, TimeUnit.SECONDS))
        assertTrue(
            "异步路径应以 YuNetClosedException 返回: ${closedError.get()}",
            closedError.get() is YuNetClosedException
        )
    }

    @Test
    fun warmUpPerformanceSmokeTest() {
        val d = detector()
        d.detectSync(sample) // 预热（首帧含解释器初始化与内存分配）
        val times = mutableListOf<Long>()
        repeat(5) {
            val t0 = System.nanoTime()
            val faces = d.detectSync(sample)
            times += (System.nanoTime() - t0) / 1_000_000
            assertTrue(faces.isNotEmpty())
        }
        val info = d.modelInfo
        Log.i(
            TAG,
            "预热后 5 次耗时(ms)=$times, 平均=${times.average()}ms, " +
                "输入=${info.inputWidth}x${info.inputHeight}, 输出=${info.outputCount}路, " +
                "anchors=${info.totalAnchors}, 名称映射=${info.mappingFromNames}"
        )
        assertTrue("平均耗时不应异常（>5s 视为失败）", times.average() < 5000)
    }

    // ---------------------------- 工具 ----------------------------

    private fun config(builder: YuNetConfig.Builder.() -> Unit = {}): YuNetConfig =
        YuNetConfig.Builder()
            .useAsset(MODEL)
            .scoreThreshold(0.6f)
            .topK(100)
            .apply(builder)
            .build()

    private fun detector(c: YuNetConfig = config()): YuNetDetector =
        YuNetDetector.create(ctx, c).also { detectors += it }

    private companion object {
        const val TAG = "YuNetDeviceTest"
        const val MODEL = "yunet_fp16.tflite"
        const val SAMPLE = "sample/largest_selfie.jpg"
        const val TIMEOUT_S = 60L
    }
}
