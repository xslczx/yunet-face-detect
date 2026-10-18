package com.yunet.face.decode

import com.yunet.face.OutputIndices
import kotlin.math.exp

/** 网络输入坐标系下的原始人脸候选（内部表示） */
internal class RawFace(
    val left: Float,
    val top: Float,
    val right: Float,
    val bottom: Float,
    val score: Float,
    /** 10 个 float（5 个关键点），无关键点时为 null */
    val landmarks: FloatArray?,
)

/**
 * 输出解码器接口。
 * 替换为其他模型（如 3 路 loc/conf/iou 的 OpenCV Zoo ONNX 版、单输出融合版）时，
 * 实现该接口并在 YuNetDetector 中注入即可，其余链路不变。
 */
internal interface OutputDecoder {
    /**
     * @param outputs 各输出张量平铺后的 FloatArray（按张量索引顺序）
     * @param inputWidth 网络输入宽
     * @param inputHeight 网络输入高
     * @return 网络输入坐标系下的人脸候选，按分数降序
     */
    fun decode(outputs: List<FloatArray>, inputWidth: Int, inputHeight: Int): List<RawFace>
}

/**
 * libfacedetection.train 导出的 12 路 YuNet 输出解码器（anchor-free 头）。
 *
 * 输出顺序（官方 README）：
 *   cls_8, cls_16, cls_32, obj_8, obj_16, obj_32, bbox_8, bbox_16, bbox_32, kps_8, kps_16, kps_32
 * 每个张量为 [1, N, C]（N = (W/s)*(H/s)，C 分别为 1/1/4/10），tflite 图内已内置 sigmoid。
 *
 * 解码公式（官方 tasks/face/codec.py + engine/priors.py，offset=0）：
 *   prior 中心: px = col*s, py = row*s（按 stride 特征图 row-major 展平）
 *   score = cls * obj
 *   cx = px + dx*s,  cy = py + dy*s
 *   w  = exp(dw)*s, h = exp(dh)*s
 *   box = (cx-w/2, cy-h/2, cx+w/2, cy+h/2)
 *   landmark_i = (px + kps_x*s, py + kps_y*s)
 * 后处理：score >= 阈值过滤 -> 硬 NMS(IoU) -> topK 截断（与官方 postprocess.py 一致）。
 */
internal class RawYuNetDecoder(
    private val strides: IntArray,
    private val scoreThreshold: Float,
    private val nmsThreshold: Float,
    private val topK: Int,
    private val nmsPreTopK: Int,
    private val applySigmoid: Boolean,
    private val indices: OutputIndices,
) : OutputDecoder {

    init {
        val s = strides.size
        require(
            indices.cls.size == s && indices.obj.size == s &&
                indices.bbox.size == s && indices.kps.size == s
        ) { "OutputIndices 各数组长度需与 strides 数量($s)一致" }
    }

    override fun decode(outputs: List<FloatArray>, inputWidth: Int, inputHeight: Int): List<RawFace> {
        require(outputs.size == strides.size * 4) {
            "期望 ${strides.size * 4} 个输出张量（cls/obj/bbox/kps × ${strides.size} strides），实际 ${outputs.size}"
        }
        val candidates = ArrayList<RawFace>()
        for (si in strides.indices) {
            val stride = strides[si]
            val cls = outputs[indices.cls[si]]
            val obj = outputs[indices.obj[si]]
            val bbox = outputs[indices.bbox[si]]
            val kps = outputs[indices.kps[si]]
            val n = cls.size
            require(obj.size == n && bbox.size == n * 4 && kps.size == n * 10) {
                "stride=$stride 输出长度不一致: cls=${cls.size}, obj=${obj.size}, " +
                    "bbox=${bbox.size}(期望 ${n * 4}), kps=${kps.size}(期望 ${n * 10})"
            }
            val fmW = inputWidth / stride
            val fmH = inputHeight / stride
            require(fmW > 0 && fmH > 0 && n == fmW * fmH) {
                "stride=$stride anchor 数不匹配: 期望 ${fmW * fmH}（${fmW}x${fmH}），实际 $n；" +
                    "请检查 inputSize 与 strides 是否与模型一致"
            }
            val sf = stride.toFloat()
            for (i in 0 until n) {
                val score = if (applySigmoid) sigmoid(cls[i]) * sigmoid(obj[i]) else cls[i] * obj[i]
                if (score < scoreThreshold) continue
                val px = (i % fmW).toFloat() * sf
                val py = (i / fmW).toFloat() * sf
                val cx = px + bbox[i * 4] * sf
                val cy = py + bbox[i * 4 + 1] * sf
                // exp 入参钳制，防止 fp16 噪声导致 Inf/NaN 扩散（与官方语义兼容的正常值远小于 20）
                val w = exp(bbox[i * 4 + 2].coerceIn(-20f, 20f)) * sf
                val h = exp(bbox[i * 4 + 3].coerceIn(-20f, 20f)) * sf
                val lms = FloatArray(10)
                for (p in 0 until 5) {
                    lms[p * 2] = px + kps[i * 10 + p * 2] * sf
                    lms[p * 2 + 1] = py + kps[i * 10 + p * 2 + 1] * sf
                }
                candidates.add(RawFace(cx - w / 2f, cy - h / 2f, cx + w / 2f, cy + h / 2f, score, lms))
            }
        }
        if (candidates.isEmpty()) return emptyList()
        // NMS 前截断，约束 O(n^2) 最坏耗时（默认阈值场景候选数通常 < 100）
        val pre = if (nmsPreTopK in 1 until candidates.size) {
            candidates.sortedByDescending { it.score }.take(nmsPreTopK)
        } else {
            candidates
        }
        val boxesArr = FloatArray(pre.size * 4)
        val scoresArr = FloatArray(pre.size)
        pre.forEachIndexed { i, f ->
            boxesArr[i * 4] = f.left
            boxesArr[i * 4 + 1] = f.top
            boxesArr[i * 4 + 2] = f.right
            boxesArr[i * 4 + 3] = f.bottom
            scoresArr[i] = f.score
        }
        val kept = Nms.nms(boxesArr, scoresArr, nmsThreshold)
        val result = kept.map { pre[it] }
        return if (topK in 1 until result.size) result.take(topK) else result
    }

    private fun sigmoid(x: Float): Float = 1f / (1f + exp(-x))
}
