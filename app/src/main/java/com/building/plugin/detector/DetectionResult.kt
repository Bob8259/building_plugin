package com.building.plugin.detector

import android.graphics.RectF

data class DetectionResult(
    val boundingBox: RectF,
    val score: Float,
    val classIndex: Int
)
