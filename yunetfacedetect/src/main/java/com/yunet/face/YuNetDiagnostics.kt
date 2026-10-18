package com.yunet.face

import android.content.Context
import android.graphics.Bitmap
import android.util.Log

/**
 * 模型自检工具。当模型文件来源不确定、输出映射存疑或检出异常时，
 * 先用 [inspect] 核对结构，再用 [selfTest] 跑通全链路。
 */
object YuNetDiagnostics {

    private const val TAG = "YuNetDiagnostics"

    /** litert-community/YuNet-Face-LiteRT 的 yunet_fp16.tflite 参考哈希 */
    private const val REFERENCE_SHA256 = "ced5f52bef6e76ad4a66d1055b2b404336ceafbae8eeac8fed6aa9c7b2e4d776"

    /**
     * 仅加载模型并输出结构报告（不执行推理）。
     * @return 人类可读的报告文本（建议 Log 打印或弹窗展示）
     */
    @Throws(YuNetException::class)
    fun inspect(context: Context, config: YuNetConfig): String {
        val engine = TfliteEngine(config)
        try {
            val info = engine.open(context.applicationContext)
            return buildString {
                appendLine("== YuNet 模型结构自检 ==")
                appendLine(info)
                appendLine(
                    "配置: score>=${config.scoreThreshold}, nms=${config.nmsThreshold}, topK=${config.topK}, " +
                        "resize=${config.resizeMode}, channel=${config.channelOrder}, " +
                        "delegate=${config.delegate}, threads=${config.numThreads}"
                )
                if (!info.mappingFromNames) {
                    appendLine(
                        "提示: 输出角色使用默认/显式索引顺序（张量名未识别）。" +
                            "若上方各输出角色与预期不符，请通过 YuNetConfig.outputIndices 显式指定。"
                    )
                }
                appendLine("参考: yunet_fp16.tflite (litert-community/YuNet-Face-LiteRT), sha256=$REFERENCE_SHA256")
            }
        } finally {
            engine.close()
        }
    }

    /**
     * 全链路自检：创建检测器并对一张图片（或内置灰度测试图）执行一次检测。
     * @param bitmap 测试图片；传 null 时用内置 640x480 灰度图（预期 0 检出，仅验证链路不崩）
     * @return 人类可读的报告文本
     */
    @Throws(YuNetException::class)
    fun selfTest(context: Context, config: YuNetConfig, bitmap: Bitmap?): String {
        var created: Bitmap? = null
        val detector = YuNetDetector.create(context, config)
        try {
            val bmp = bitmap ?: Bitmap.createBitmap(640, 480, Bitmap.Config.ARGB_8888)
                .apply { eraseColor(0xFF808080.toInt()) }
                .also { created = it }
            val t0 = System.currentTimeMillis()
            val faces = detector.detectSync(bmp)
            val ms = System.currentTimeMillis() - t0
            return buildString {
                appendLine("== YuNet 运行自检 ==")
                appendLine("输入: ${bmp.width}x${bmp.height}, 首帧耗时 ${ms}ms（含初始化预热）")
                appendLine("检出人脸: ${faces.size}")
                faces.forEachIndexed { i, f ->
                    appendLine(
                        "  #$i score=${"%.3f".format(f.score)} box=[${f.box.left.toInt()}," +
                            "${f.box.top.toInt()},${f.box.right.toInt()},${f.box.bottom.toInt()}] " +
                            "eyeAngle=${f.eyeAngleDegrees()}deg"
                    )
                }
                if (faces.isEmpty()) {
                    appendLine()
                    appendLine("0 检出的排查顺序:")
                    appendLine("  1) resizeMode 切换 LETTERBOX <-> STRETCH 对比")
                    appendLine("  2) channelOrder 切换 BGR <-> RGB")
                    appendLine("  3) 降低 scoreThreshold 到 0.3，观察是否出现低分框")
                    appendLine("  4) 核对模型文件 sha256=$REFERENCE_SHA256")
                    appendLine("  5) 若分数恒为 0 或 1，尝试 applySigmoidToScores=true")
                    appendLine("  6) 用 YuNetDiagnostics.inspect 核对输出张量角色映射")
                }
                appendLine()
                appendLine(
                    "landmark 顺序验证: 用一张正脸照片检测，检查 index0/index1 是否构成双眼、" +
                        "index2 是否鼻尖；命名约定（labelv2/SCRFD 顺序）见 Face 类注释。"
                )
            }
        } finally {
            runCatching { detector.close() }
            runCatching { created?.recycle() }
        }
    }
}
