package com.yunet.face.decode

import kotlin.math.max
import kotlin.math.min

/**
 * 硬 NMS。语义与 libfacedetection.train 的 engine/nms.py 一致：
 * 按分数降序扫描，保留当前最高分框，抑制与其 IoU > 阈值的其余框。
 * 单类检测场景（人脸）等价于官方 batched_nms。
 */
internal object Nms {

    /**
     * @param boxes 长度 4n 的 x1y1x2y2 平铺数组
     * @param scores 长度 n 的分数
     * @param iouThreshold 抑制阈值：IoU 大于该值即被抑制
     * @return 保留的索引数组，按分数降序
     */
    fun nms(boxes: FloatArray, scores: FloatArray, iouThreshold: Float): IntArray {
        val n = scores.size
        if (n == 0) return IntArray(0)
        if (n == 1) return intArrayOf(0)
        // 分数降序；同分保持原索引顺序，保证结果稳定可复现
        val sorted = IntArray(n) { it }
            .sortedWith(compareByDescending<Int> { scores[it] }.thenBy { it })
            .toIntArray()
        val suppressed = BooleanArray(n)
        val keep = ArrayList<Int>(minOf(64, n))
        for (i in 0 until n) {
            val idx = sorted[i]
            if (suppressed[idx]) continue
            keep.add(idx)
            val ax1 = boxes[idx * 4]
            val ay1 = boxes[idx * 4 + 1]
            val ax2 = boxes[idx * 4 + 2]
            val ay2 = boxes[idx * 4 + 3]
            val aArea = max(0f, ax2 - ax1) * max(0f, ay2 - ay1)
            for (j in i + 1 until n) {
                val other = sorted[j]
                if (suppressed[other]) continue
                val interW = min(ax2, boxes[other * 4 + 2]) - max(ax1, boxes[other * 4])
                if (interW <= 0f) continue
                val interH = min(ay2, boxes[other * 4 + 3]) - max(ay1, boxes[other * 4 + 1])
                if (interH <= 0f) continue
                val inter = interW * interH
                val bArea = max(0f, boxes[other * 4 + 2] - boxes[other * 4]) *
                    max(0f, boxes[other * 4 + 3] - boxes[other * 4 + 1])
                val union = aArea + bArea - inter
                if (union <= 0f) continue
                if (inter / union > iouThreshold) suppressed[other] = true
            }
        }
        return keep.toIntArray()
    }
}
