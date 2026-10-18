package com.yunet.face

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs

/**
 * EXIF 镜像/旋转几何与关键点一致性校验的 JVM 单元测试。
 * 运行: ./gradlew :yunetfacedetect:testDebugUnitTest
 */
class ExifMirrorTest {

    // ---------------- CoordinateMapper（转正逆映射，含镜像） ----------------

    /**
     * 构造“网络坐标已与竖直坐标重合（letterbox scale=1、pad=0）”的映射器，
     * 用于对 EXIF 逆变换做正向/逆向往返校验。
     */
    private fun cycleMapper(
        rotationDeg: Int,
        mirrored: Boolean,
        decodedW: Int,
        decodedH: Int,
    ): CoordinateMapper {
        val swap = rotationDeg == 90 || rotationDeg == 270
        return CoordinateMapper(
            resizeMode = ResizeMode.LETTERBOX,
            scale = 1f, scaleX = 1f, scaleY = 1f, padX = 0f, padY = 0f,
            rotationDeg = rotationDeg.toFloat(),
            uprightW = if (swap) decodedH else decodedW,
            uprightH = if (swap) decodedW else decodedH,
            finalW = decodedW,
            finalH = decodedH,
            finalScale = 1f,
            mirrored = mirrored,
        )
    }

    /**
     * 转正正变换（androidx 语义：先水平镜像、后顺时针旋转），
     * 用于生成“已知应映射回原点”的竖直坐标。
     */
    private fun forward(
        x: Float, y: Float, w: Int, h: Int, rotationDeg: Int, mirrored: Boolean,
    ): FloatArray {
        var px = if (mirrored) (w - x) else x
        var py = y
        return when (rotationDeg) {
            90 -> floatArrayOf(h - py, px)
            180 -> floatArrayOf(w - px, h - py)
            270 -> floatArrayOf(py, w - px)
            else -> floatArrayOf(px, py)
        }
    }

    @Test
    fun mapper_cycle_roundTrip_allOrientations() {
        val w = 100
        val h = 60
        // 解码坐标取远离边缘的内部点，避免钳制干扰
        val decodedPoints = listOf(
            floatArrayOf(10f, 20f), floatArrayOf(90f, 40f), floatArrayOf(33.5f, 21.25f),
        )
        val cases = listOf(
            0 to false,   // orientation 1
            0 to true,    // orientation 2（镜像）
            180 to false, // orientation 3
            180 to true,  // orientation 4（镜像+180，等效垂直翻转）
            90 to false,  // orientation 6
            270 to true,  // orientation 5（镜像+270）
            90 to true,   // orientation 7（镜像+90）
            270 to false, // orientation 8
        )
        for ((rot, mir) in cases) {
            val mapper = cycleMapper(rot, mir, w, h)
            for (p in decodedPoints) {
                val upright = forward(p[0], p[1], w, h, rot, mir)
                val back = mapper.mapPoint(upright[0], upright[1])
                assertTrue(
                    "rot=$rot mirrored=$mir 端点=${p.toList()} -> upright=${upright.toList()}",
                    abs(back[0] - p[0]) < 1e-3 && abs(back[1] - p[1]) < 1e-3,
                )
            }
        }
    }

    @Test
    fun mapper_mirroredFlagChannelsThroughBuilderDefaults() {
        // 默认构造（无镜像）应与显式 mirrored=false 一致：orientation 2 旧行为（不翻转）
        val w = 100
        val h = 60
        val mapper = CoordinateMapper(
            resizeMode = ResizeMode.LETTERBOX,
            scale = 1f, scaleX = 1f, scaleY = 1f, padX = 0f, padY = 0f,
            rotationDeg = 0f,
            uprightW = w, uprightH = h,
            finalW = w, finalH = h,
            finalScale = 1f,
        )
        // 镜像后转正坐标 (W−x, y) 在旧实现（忽略镜像）下会原样返回
        val back = mapper.mapPoint(w - 10f, 20f)
        assertEquals(w - 10f, back[0], 1e-6f)
        assertEquals(20f, back[1], 1e-6f)
    }

    // ---------------- ExifTransform ----------------

    @Test
    fun exifTransform_normalizesAndIsPure() {
        assertEquals(90, ExifTransform.of(90, false).rotationDegrees)
        assertEquals(0, ExifTransform.of(360, false).rotationDegrees)
        assertEquals(270, ExifTransform.of(-90, false).rotationDegrees)
        assertTrue(ExifTransform.of(0, true).mirrored)
        assertEquals(ExifTransform.IDENTITY.toString(), ExifTransform.of(0, false).toString())
    }

    @Test
    fun exifTransform_uprightSizeSwapsForQuarterTurns() {
        assertTrue(ExifTransform.of(90, false).uprightSize(100, 60).contentEquals(intArrayOf(60, 100)))
        assertTrue(ExifTransform.of(180, true).uprightSize(100, 60).contentEquals(intArrayOf(100, 60)))
    }

    // ---------------- Face 关键点一致性 ----------------

    private val frontal = floatArrayOf(
        30f, 20f,   // 左眼
        70f, 20f,   // 右眼
        50f, 38f,   // 鼻尖（双眼中点与嘴中点连线上）
        35f, 60f,   // 左嘴角
        65f, 60f,   // 右嘴角
    )

    @Test
    fun landmarks_consistentFrontalAndSlightProfile() {
        assertTrue(Face.landmarksConsistent(frontal))
        // 中等侧脸：鼻尖水平偏移但仍在 0.85 容差内，不误伤
        val slightYaw = frontal.copyOf()
        slightYaw[4] = 80f; slightYaw[5] = 42f
        assertTrue(Face.landmarksConsistent(slightYaw))
    }

    @Test
    fun landmarks_invalidInputs() {
        assertFalse(Face.landmarksConsistent(null))
        assertFalse(Face.landmarksConsistent(FloatArray(4)))
        // 双眼重合
        val deadEyes = frontal.copyOf().also { it[2] = 30f; it[3] = 20f }
        assertFalse(Face.landmarksConsistent(deadEyes))
        // 嘴距相对眼距异常（角色疑似互换）
        val swapped = frontal.copyOf().also { it[6] = 50f; it[7] = 39f; it[8] = 50.2f; it[9] = 39f }
        assertFalse(Face.landmarksConsistent(swapped))
    }

    @Test
    fun landmarks_clearlyBrokenGeometryFlagged() {
        // 鼻尖远离“双眼中点-嘴中点”连线（垂距超 0.85×线段长）
        val offNose = frontal.copyOf().also { it[4] = 95f; it[5] = 40f }
        assertFalse(Face.landmarksConsistent(offNose))
        // 鼻尖不在段延长范围内
        val outOfRange = frontal.copyOf().also { it[5] = 175f }
        assertFalse(Face.landmarksConsistent(outOfRange))
    }
}
