package com.yunet.face

import com.yunet.face.decode.Nms
import com.yunet.face.decode.RawYuNetDecoder
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * 纯 Kotlin 核心逻辑（NMS / anchor-free 解码 / 坐标映射）的 JVM 单元测试。
 * 运行: ./gradlew :yunetfacedetect:testDebugUnitTest
 */
class YuNetCoreTest {

    // ---------------- NMS ----------------

    @Test
    fun nms_suppressesOverlapKeepsDisjoint() {
        // 两个高度重叠框（IoU≈0.82 > 0.45）+ 一个远离框
        val boxes = floatArrayOf(
            0f, 0f, 10f, 10f,
            0.5f, 0.5f, 10.5f, 10.5f,
            50f, 50f, 60f, 60f,
        )
        val scores = floatArrayOf(0.9f, 0.85f, 0.8f)
        assertArrayEquals(intArrayOf(0, 2), Nms.nms(boxes, scores, 0.45f))
    }

    @Test
    fun nms_emptyAndSingle() {
        assertArrayEquals(intArrayOf(), Nms.nms(floatArrayOf(), floatArrayOf(), 0.45f))
        assertArrayEquals(
            intArrayOf(0),
            Nms.nms(floatArrayOf(0f, 0f, 1f, 1f), floatArrayOf(0.5f), 0.45f)
        )
    }

    // ---------------- 解码器 ----------------

    private fun singleStrideDecoder(
        scoreThreshold: Float = 0.5f,
        applySigmoid: Boolean = false,
    ) = RawYuNetDecoder(
        strides = intArrayOf(8),
        scoreThreshold = scoreThreshold,
        nmsThreshold = 0.45f,
        topK = 0,
        nmsPreTopK = 0,
        applySigmoid = applySigmoid,
        indices = OutputIndices(intArrayOf(0), intArrayOf(1), intArrayOf(2), intArrayOf(3)),
    )

    @Test
    fun decode_singleCandidateFormulas() {
        // 输入 16x16, stride 8 -> 2x2 = 4 anchors；唯一高分候选放在 cell(row=0, col=1)，prior=(8,0)
        val cls = floatArrayOf(0.1f, 0.9f, 0.1f, 0.1f)
        val obj = floatArrayOf(0.9f, 0.8f, 0.9f, 0.9f)
        val bbox = FloatArray(16)
        val kps = FloatArray(40)
        val i = 1
        bbox[i * 4] = 0.5f           // cx = 8 + 0.5*8 = 12
        bbox[i * 4 + 1] = 0.25f      // cy = 0 + 0.25*8 = 2
        bbox[i * 4 + 2] = 0.6931472f // w  = exp(dw)*8 = 2*8 = 16
        bbox[i * 4 + 3] = 1.3862944f // h  = exp(dh)*8 = 4*8 = 32
        kps[i * 10] = 1f             // l0x = 8 + 1*8 = 16
        kps[i * 10 + 1] = 0f         // l0y = 0
        kps[i * 10 + 2] = -1f        // l1x = 8 - 8 = 0
        kps[i * 10 + 3] = 1f         // l1y = 0 + 8 = 8

        val faces = singleStrideDecoder().decode(listOf(cls, obj, bbox, kps), 16, 16)
        assertEquals(1, faces.size)
        val f = faces[0]
        assertEquals(12f - 8f, f.left, 1e-4f)
        assertEquals(2f - 16f, f.top, 1e-4f)
        assertEquals(12f + 8f, f.right, 1e-4f)
        assertEquals(2f + 16f, f.bottom, 1e-4f)
        assertEquals(0.72f, f.score, 1e-4f)
        assertEquals(16f, f.landmarks!![0], 1e-4f)
        assertEquals(0f, f.landmarks!![1], 1e-4f)
        assertEquals(0f, f.landmarks!![2], 1e-4f)
        assertEquals(8f, f.landmarks!![3], 1e-4f)
    }

    @Test
    fun decode_filtersByScore() {
        val cls = floatArrayOf(0.1f, 0.1f, 0.1f, 0.1f)
        val obj = floatArrayOf(0.9f, 0.9f, 0.9f, 0.9f)
        val faces = singleStrideDecoder(scoreThreshold = 0.5f)
            .decode(listOf(cls, obj, FloatArray(16), FloatArray(40)), 16, 16)
        assertTrue(faces.isEmpty())
    }

    @Test
    fun decode_applySigmoid() {
        // logits 2.0 -> sigmoid ≈ 0.88；0.88*0.88 ≈ 0.776 >= 0.5
        val cls = floatArrayOf(2f, -2f, -2f, -2f)
        val obj = floatArrayOf(2f, 2f, 2f, 2f)
        val faces = singleStrideDecoder(applySigmoid = true)
            .decode(listOf(cls, obj, FloatArray(16), FloatArray(40)), 16, 16)
        assertEquals(1, faces.size)
        assertEquals(0.776f, faces[0].score, 1e-2f)
    }

    @Test(expected = IllegalArgumentException::class)
    fun decode_anchorCountMismatchThrows() {
        // 输出按 5 个 anchor 组织，但 16x16/stride8 期望 4 个
        singleStrideDecoder().decode(
            listOf(FloatArray(5), FloatArray(5), FloatArray(20), FloatArray(50)),
            16,
            16,
        )
    }

    // ---------------- 坐标映射 ----------------

    private fun mapper(
        mode: ResizeMode,
        scale: Float = 1f,
        scaleX: Float = 1f,
        scaleY: Float = 1f,
        rotationDeg: Float = 0f,
        uprightW: Int,
        uprightH: Int,
        finalW: Int,
        finalH: Int,
        finalScale: Float = 1f,
    ) = CoordinateMapper(
        resizeMode = mode,
        scale = scale,
        scaleX = scaleX,
        scaleY = scaleY,
        padX = 0f,
        padY = 0f,
        rotationDeg = rotationDeg,
        uprightW = uprightW,
        uprightH = uprightH,
        finalW = finalW,
        finalH = finalH,
        finalScale = finalScale,
    )

    @Test
    fun mapper_letterbox() {
        // 竖直图 1280x1280，letterbox 到 640x640 -> scale 0.5；net(320,320) -> 竖直(640,640)
        val p = mapper(ResizeMode.LETTERBOX, scale = 0.5f, uprightW = 1280, uprightH = 1280, finalW = 1280, finalH = 1280)
            .mapPoint(320f, 320f)
        assertEquals(640f, p[0], 1e-3f)
        assertEquals(640f, p[1], 1e-3f)
    }

    @Test
    fun mapper_stretch() {
        // 竖直图 640x160，stretch 到 640x640 -> scaleX=1, scaleY=4
        // net(320,160) 对应竖直(320, 160/4=40)
        val p = mapper(ResizeMode.STRETCH, scaleX = 1f, scaleY = 4f, uprightW = 640, uprightH = 160, finalW = 640, finalH = 640)
            .mapPoint(320f, 160f)
        assertEquals(320f, p[0], 1e-3f)
        assertEquals(40f, p[1], 1e-3f)
    }

    @Test
    fun mapper_inverseRotation90() {
        // b0 = 50x100，顺时针转 90 成 upright 100x50；upright(10,20) -> b0(20, 90)
        val p = mapper(ResizeMode.LETTERBOX, rotationDeg = 90f, uprightW = 100, uprightH = 50, finalW = 50, finalH = 100)
            .mapPoint(10f, 20f)
        assertEquals(20f, p[0], 1e-3f)
        assertEquals(90f, p[1], 1e-3f)
    }

    @Test
    fun mapper_inverseRotation180() {
        val p = mapper(ResizeMode.LETTERBOX, rotationDeg = 180f, uprightW = 50, uprightH = 100, finalW = 50, finalH = 100)
            .mapPoint(10f, 20f)
        assertEquals(40f, p[0], 1e-3f)
        assertEquals(80f, p[1], 1e-3f)
    }

    @Test
    fun mapper_inverseRotation270() {
        // b0 = 100x50，顺时针转 270（即逆 90）成 upright 50x100；upright(10,20) -> b0(80, 10)
        val p = mapper(ResizeMode.LETTERBOX, rotationDeg = 270f, uprightW = 50, uprightH = 100, finalW = 100, finalH = 50)
            .mapPoint(10f, 20f)
        assertEquals(80f, p[0], 1e-3f)
        assertEquals(10f, p[1], 1e-3f)
    }

    @Test
    fun mapper_boxIsBoundingRectOfCorners() {
        val m = mapper(ResizeMode.LETTERBOX, rotationDeg = 90f, uprightW = 100, uprightH = 50, finalW = 50, finalH = 100)
        val box = m.mapBox(0f, 0f, 10f, 10f)
        // 四角旋转后外包矩形
        assertEquals(0f, box[0], 1e-3f)   // left  = min y = 0
        assertEquals(90f, box[1], 1e-3f)  // top   = 100 - max x = 100-10
        assertEquals(10f, box[2], 1e-3f)  // right = max y = 10
        assertEquals(100f, box[3], 1e-3f) // bottom = 100 - min x = 100
    }
}
