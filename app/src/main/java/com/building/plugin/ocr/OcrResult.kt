package com.building.plugin.ocr

import android.graphics.RectF

/**
 * A single recognized text line produced by [PpOcrEngine].
 *
 * @param text recognized text content of the line
 * @param score detection confidence (mean DB probability of the bounding box, 0..1)
 * @param confidence recognition confidence (mean CTC score of recognized characters, 0..1)
 * @param box bounding box in original image coordinates
 */
data class OcrResult(
    val text: String,
    val score: Float,
    val confidence: Float,
    val box: RectF
)

