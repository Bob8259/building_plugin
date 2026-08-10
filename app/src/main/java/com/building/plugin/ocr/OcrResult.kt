package com.building.plugin.ocr

import android.graphics.RectF

/**
 * A single recognized text line produced by [PpOcrEngine].
 *
 * @param text recognized text content of the line
 * @param score detection confidence (mean DB probability of the text box, 0–1)
 * @param box bounding box in original image pixel coordinates
 */
data class OcrResult(
    val text: String,
    val score: Float,
    val box: RectF
)
