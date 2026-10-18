package com.yunet.face.demo

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.RectF
import android.graphics.Typeface
import android.util.AttributeSet
import android.view.View
import com.yunet.face.Face

/**
 * 把 [Face] 结果（原图像素坐标）叠加绘制到等比缩放显示的原图上。
 *
 * 显示区域 = 图片按 fit-center 缩放后的矩形，人脸框 / 5 关键点 / 双眼连线
 * 通过同一个 [Matrix] 映射，因而与库返回的坐标一一对应（可直接目视核对
 * 坐标回溯是否正确）。
 */
class FaceOverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0,
) : View(context, attrs, defStyleAttr) {

    private var bitmap: Bitmap? = null
    private var faces: List<Face> = emptyList()
    private val imageMatrix = Matrix()

    private val bitmapPaint = Paint(Paint.FILTER_BITMAP_FLAG or Paint.DITHER_FLAG)
    private val boxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = dp(2f)
        color = 0xFF00E676.toInt()
    }
    private val eyeLinePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = dp(1.5f)
        color = 0xCC00BCD4.toInt()
    }
    private val pointPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
        color = 0xFFFF5252.toInt()
    }
    private val labelBgPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
        color = 0xCC000000.toInt()
    }
    private val labelPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textSize = dp(12f)
        typeface = Typeface.MONOSPACE
    }
    private val hintPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0xFF9E9E9E.toInt()
        textSize = dp(14f)
    }

    /** 设置底图（会清空上一次的结果） */
    fun setImage(bmp: Bitmap?) {
        bitmap = bmp
        faces = emptyList()
        invalidate()
    }

    /** 设置检测结果 */
    fun setFaces(list: List<Face>) {
        faces = list
        invalidate()
    }

    override fun onSizeChanged(w: Int, h: Int, oldw: Int, oldh: Int) {
        super.onSizeChanged(w, h, oldw, oldh)
        updateMatrix(w, h)
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val bmp = bitmap
        if (bmp == null || bmp.isRecycled) {
            canvas.drawText(context.getString(R.string.hint_pick_image), dp(12f), dp(24f), hintPaint)
            return
        }
        updateMatrix(width, height)
        canvas.drawBitmap(bmp, imageMatrix, bitmapPaint)

        for (face in faces) {
            val box = RectF(face.box)
            imageMatrix.mapRect(box)
            canvas.drawRect(box, boxPaint)

            val lm = face.landmarks
            if (lm != null) {
                // 双眼连线（0/1 号点，解剖学左右：正脸时 0 号点偏图右）
                val eyeA = mapPoint(lm[0], lm[1])
                val eyeB = mapPoint(lm[2], lm[3])
                canvas.drawLine(eyeA[0], eyeA[1], eyeB[0], eyeB[1], eyeLinePaint)
                for (i in 0 until Face.LANDMARK_COUNT) {
                    val p = mapPoint(lm[i * 2], lm[i * 2 + 1])
                    canvas.drawCircle(p[0], p[1], dp(3f), pointPaint)
                }
            }

            // 小脸不画分数标签，避免多人合照时文字糊成一片
            if (box.width() >= dp(36f)) {
                val label = "%.2f %+.1f°".format(face.score, face.eyeAngleDegrees())
                val textW = labelPaint.measureText(label)
                val textH = labelPaint.textSize
                val left = box.left.coerceIn(0f, (width - textW - dp(4f)).coerceAtLeast(0f))
                val top = (box.top - textH - dp(4f)).coerceAtLeast(0f)
                canvas.drawRect(left, top, left + textW + dp(4f), top + textH + dp(3f), labelBgPaint)
                canvas.drawText(label, left + dp(2f), top + textH, labelPaint)
            }
        }
    }

    private fun mapPoint(x: Float, y: Float): FloatArray {
        val p = floatArrayOf(x, y)
        imageMatrix.mapPoints(p)
        return p
    }

    /** 图片 -> 视图 的 fit-center 映射 */
    private fun updateMatrix(w: Int, h: Int) {
        val bmp = bitmap ?: return
        if (w <= 0 || h <= 0 || bmp.width <= 0 || bmp.height <= 0) return
        val scale = minOf(w.toFloat() / bmp.width, h.toFloat() / bmp.height)
        imageMatrix.reset()
        imageMatrix.postScale(scale, scale)
        imageMatrix.postTranslate((w - bmp.width * scale) / 2f, (h - bmp.height * scale) / 2f)
    }

    private fun dp(v: Float): Float = v * resources.displayMetrics.density
}
