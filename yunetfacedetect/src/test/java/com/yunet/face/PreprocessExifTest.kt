package com.yunet.face

import android.graphics.Bitmap
import android.graphics.Color
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.annotation.Config
import org.robolectric.annotation.GraphicsMode

/**
 * Preprocess 镜像/旋转转正的像素级 JVM 测试（Robolectric native graphics）。
 * 运行: ./gradlew :yunetfacedetect:testDebugUnitTest
 *
 * 覆盖点（此前只能靠真机 connected 测试）：
 * - 无 EXIF 直通、纯镜像（orientation 2）、纯旋转（6/8）、组合（4/5/7）的像素正确性；
 * - Result 的 mirrored/rotationDeg/uprightW/uprightH 元信息；
 * - “先镜像、后旋转”顺序（与 androidx exifinterface 语义一致）。
 */
@RunWith(AndroidJUnit4::class)
@Config(sdk = [34])
@GraphicsMode(GraphicsMode.Mode.NATIVE)
class PreprocessExifTest {

    private fun preprocessor(netW: Int = 8, netH: Int = 6) = Preprocessor(
        config = YuNetConfig.Builder().useAsset("dummy.tflite").build(),
        channels = 3,
        layout = InputLayout.NCHW,
        netW = netW,
        netH = netH,
    )

    /**
     * 不对称测试位图：8x6（恰好等于网络输入 → letterbox 不缩放不补边），
     * 像素颜色由坐标决定，便于任意方向断言。
     */
    private fun src(w: Int = 8, h: Int = 6): Bitmap {
        val bmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        for (y in 0 until h) for (x in 0 until w) {
            bmp.setPixel(x, y, Color.rgb(10 + x * 20, 10 + y * 30, 0))
        }
        return bmp
    }

    /** 把 Result 的 BGR/NCHW 像素缓冲转回 Bitmap（便于直观断言） */
    private fun Preprocessor.Result.toBitmap(w: Int, h: Int): Bitmap {
        val bmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        val n = w * h
        for (y in 0 until h) for (x in 0 until w) {
            val i = y * w + x
            val b = pixels[0 * n + i].toInt()   // BGR：0=B
            val g = pixels[1 * n + i].toInt()
            val r = pixels[2 * n + i].toInt()
            bmp.setPixel(x, y, Color.rgb(r, g, b))
        }
        return bmp
    }

    /** 断言 out(ox,oy) == src(sx,sy)（比较 R/G 两个非零通道） */
    private fun assertSamePixel(out: Bitmap, ox: Int, oy: Int, srcBitmap: Bitmap, sx: Int, sy: Int) {
        val a = out.getPixel(ox, oy)
        val b = srcBitmap.getPixel(sx, sy)
        assertTrue(
            "out($ox,$oy)=${Color.red(a)},${Color.green(a)} 应等于 src($sx,$sy)=${Color.red(b)},${Color.green(b)}",
            Color.red(a) == Color.red(b) && Color.green(a) == Color.green(b),
        )
    }

    @Test
    fun noExif_identityPassThrough() {
        val result = preprocessor().process(src(), ExifTransform.IDENTITY)
        val n = result.uprightH * result.uprightW
        fun p(ch: Int, x: Int, y: Int): Float = result.pixels[ch * n + y * result.uprightW + x]
        // (0,0)：R=10 G=10 B=0；BGR → ch0=B=0, ch1=G=10, ch2=R=10
        assertTrue(p(0, 0, 0) == 0f)
        assertTrue(p(1, 0, 0) == 10f)
        assertTrue(p(2, 0, 0) == 10f)
        // (7,5)：R=150 G=160
        assertTrue(p(2, 7, 5) == 150f)
        assertTrue(p(1, 7, 5) == 160f)
    }

    @Test
    fun orientation2_mirrorOnly() {
        val s = src()
        val result = preprocessor().process(s, ExifTransform.of(0, true))
        assertTrue(result.mirrored)
        assertTrue(result.rotationDeg == 0f)
        assertTrue(result.uprightW == 8 && result.uprightH == 6)
        // 水平镜像：out(x,y) = src(w-1-x, y)
        val upright = result.toBitmap(8, 6)
        assertSamePixel(upright, 0, 0, s, 7, 0)
        assertSamePixel(upright, 7, 5, s, 0, 5)
        assertSamePixel(upright, 3, 2, s, 4, 2)
    }

    @Test
    fun orientation4_mirrorThen180_isVerticalFlip() {
        val s = src()
        val result = preprocessor().process(s, ExifTransform.of(180, true))
        assertTrue(result.mirrored)
        // 先镜像后旋转 180° == 垂直翻转：out(x,y) = src(x, h-1-y)
        val upright = result.toBitmap(8, 6)
        assertSamePixel(upright, 0, 0, s, 0, 5)
        assertSamePixel(upright, 7, 5, s, 7, 0)
        assertSamePixel(upright, 3, 2, s, 3, 3)
    }

    @Test
    fun orientation6_rotationOnly() {
        val s = src()
        // 旋转后竖直图为 6x8；net 8x8 时 letterbox scale=1（右侧 2 列为 pad），无插值
        val result = preprocessor(netW = 8, netH = 8).process(s, ExifTransform.of(90, false))
        assertFalse(result.mirrored)
        assertTrue(result.rotationDeg == 90f)
        assertTrue(result.uprightW == 6 && result.uprightH == 8)
        val n = 8 * 8
        // 顺时针 90°取值：upright(x',y') = src(y', h-1-x')（内容贴左上）
        for (probe in listOf(0 to 0, 5 to 7, 2 to 3)) {
            val (x, y) = probe
            val r = result.pixels[2 * n + y * 8 + x].toInt()
            val g = result.pixels[1 * n + y * 8 + x].toInt()
            val sx = y
            val sy = 5 - x
            assertTrue(
                "out($x,$y)=$r,$g 应为 src($sx,$sy) R=${10 + sx * 20} G=${10 + sy * 30}",
                r == 10 + sx * 20 && g == 10 + sy * 30,
            )
        }
        // 右侧 x>=6 为 letterbox 补边（padValue=0 → 全黑）
        assertTrue(result.pixels[2 * n + 0 * 8 + 7] == 0f)
        assertTrue(result.pixels[1 * n + 0 * 8 + 7] == 0f)
    }

    @Test
    fun orientation5_mirrorThenRotate270_isTranspose() {
        val s = src()
        val result = preprocessor(netW = 8, netH = 8).process(s, ExifTransform.of(270, true))
        assertTrue(result.mirrored)
        assertTrue(result.uprightW == 6 && result.uprightH == 8)
        val n = 8 * 8
        // 先镜像后转 270° == 主对角线转置：upright(x',y') = src(y', x')
        for (probe in listOf(5 to 2, 1 to 7, 0 to 0)) {
            val (x, y) = probe
            val r = result.pixels[2 * n + y * 8 + x].toInt()
            val g = result.pixels[1 * n + y * 8 + x].toInt()
            assertTrue(
                "out($x,$y)=$r,$g 应为 src($y,$x)",
                r == 10 + y * 20 && g == 10 + x * 30,
            )
        }
    }

    @Test
    fun orientation7_mirrorThenRotate90_isAntiTranspose() {
        val s = src()
        val result = preprocessor(netW = 8, netH = 8).process(s, ExifTransform.of(90, true))
        assertTrue(result.mirrored)
        assertTrue(result.uprightW == 6 && result.uprightH == 8)
        val n = 8 * 8
        // 先镜像后转 90° == 反对角线：upright(x',y') = src(w-1-y', h-1-x')
        for (probe in listOf(0 to 0, 5 to 7, 2 to 3)) {
            val (x, y) = probe
            val r = result.pixels[2 * n + y * 8 + x].toInt()
            val g = result.pixels[1 * n + y * 8 + x].toInt()
            val sx = 7 - y
            val sy = 5 - x
            assertTrue(
                "out($x,$y)=$r,$g 应为 src($sx,$sy)",
                r == 10 + sx * 20 && g == 10 + sy * 30,
            )
        }
    }
}
