package com.yunet.face.demo

import android.graphics.BitmapFactory
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import java.io.File
import kotlinx.coroutines.runBlocking
import org.junit.Test
import org.junit.runner.RunWith

/** [TEMP-DEBUG] 人脸裁剪链路探针：复现 /data/... 场景并输出各环节尺寸 */
@RunWith(AndroidJUnit4::class)
class CropDebugProbeTest {

    private val TAG = "YuNetCropProbe"

    @Test
    fun probeCropPipeline() {
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        val tmp = File(ctx.cacheDir, "probe_selfie.jpg")
        InstrumentationRegistry.getInstrumentation().context.assets.open("sample/largest_selfie.jpg")
            .use { input -> tmp.outputStream().use { input.copyTo(it) } }
        Log.i(TAG, "input=${tmp.absolutePath} size=${tmp.length()}B")

        val out = runBlocking {
            ArchiveFaceCrop.crop(ctx, tmp.absolutePath) { m -> Log.i(TAG, "REPORT: $m") }
        }
        Log.i(TAG, "out=$out")
        if (out != null) {
            val bmp = BitmapFactory.decodeFile(out)
            Log.i(TAG, "outDim=${bmp?.width}x${bmp?.height}")
            bmp?.recycle()
        }
        tmp.delete()
    }
}
