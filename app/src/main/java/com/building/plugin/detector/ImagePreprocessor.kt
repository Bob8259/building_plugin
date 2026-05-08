package com.building.plugin.detector

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.os.Build
import androidx.core.graphics.createBitmap

/**
 * Shared image preprocessing utilities for ONNX runtime.
 */
internal object ImagePreprocessor {

    /**
     * Resizes bitmap to model input size with letterbox padding (black fill),
     * preserving aspect ratio. Returns the padded bitmap, scale factor, and offsets.
     */
    fun resizeWithPadding(
        src: Bitmap,
        targetWidth: Int,
        targetHeight: Int
    ): Triple<Bitmap, Float, Pair<Float, Float>> {
        val srcWidth = src.width.toFloat()
        val srcHeight = src.height.toFloat()

        val scale = (targetWidth.toFloat() / srcWidth).coerceAtMost(targetHeight.toFloat() / srcHeight)
        val newWidth = srcWidth * scale
        val newHeight = srcHeight * scale
        val offsetX = (targetWidth - newWidth) / 2f
        val offsetY = (targetHeight - newHeight) / 2f

        val output = createBitmap(targetWidth, targetHeight)
        val canvas = Canvas(output)
        canvas.drawColor(Color.BLACK)

        val matrix = Matrix()
        matrix.postScale(scale, scale)
        matrix.postTranslate(offsetX, offsetY)

        val paint = Paint().apply { isFilterBitmap = true }
        canvas.drawBitmap(src, matrix, paint)

        return Triple(output, scale, Pair(offsetX, offsetY))
    }

    /**
     * Converts bitmap to NCHW float array normalized to [0, 1] for ONNX Runtime.
     */
    fun convertBitmapToFloatArrayNCHW(bitmap: Bitmap, inputWidth: Int, inputHeight: Int): FloatArray {
        val softwareBitmap = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O &&
            bitmap.config == Bitmap.Config.HARDWARE
        ) {
            bitmap.copy(Bitmap.Config.ARGB_8888, false)
        } else {
            bitmap
        }

        val pixels = IntArray(inputWidth * inputHeight)
        softwareBitmap.getPixels(pixels, 0, softwareBitmap.width, 0, 0, softwareBitmap.width, softwareBitmap.height)

        val floatArray = FloatArray(3 * inputHeight * inputWidth)
        val channelSize = inputHeight * inputWidth

        for (i in pixels.indices) {
            val value = pixels[i]
            floatArray[i] = (value shr 16 and 0xFF) / 255.0f              // R
            floatArray[channelSize + i] = (value shr 8 and 0xFF) / 255.0f  // G
            floatArray[2 * channelSize + i] = (value and 0xFF) / 255.0f    // B
        }

        if (softwareBitmap != bitmap) {
            softwareBitmap.recycle()
        }
        return floatArray
    }

    /**
     * Parses raw model output into DetectionResult list, mapping coordinates
     * back to original image space.
     */
    fun parseDetections(
        output: Array<FloatArray>,
        inputWidth: Int,
        inputHeight: Int,
        scale: Float,
        offX: Float,
        offY: Float,
        threshold: Float,
        normalizedCoords: Boolean = true
    ): List<DetectionResult> {
        val detections = mutableListOf<DetectionResult>()
        for (detection in output) {
            // detection: [x1, y1, x2, y2, score, class]
            val score = detection[4]
            if (score > threshold) {
                val x1: Float
                val y1: Float
                val x2: Float
                val y2: Float
                if (normalizedCoords) {
                    // Model outputs normalized coords (0–1), scale to pixel space
                    x1 = (detection[0] * inputWidth - offX) / scale
                    y1 = (detection[1] * inputHeight - offY) / scale
                    x2 = (detection[2] * inputWidth - offX) / scale
                    y2 = (detection[3] * inputHeight - offY) / scale
                } else {
                    // Model outputs pixel-space coords, just remove padding and rescale
                    x1 = (detection[0] - offX) / scale
                    y1 = (detection[1] - offY) / scale
                    x2 = (detection[2] - offX) / scale
                    y2 = (detection[3] - offY) / scale
                }
                val classIdx = detection[5]

                detections.add(
                    DetectionResult(
                        boundingBox = android.graphics.RectF(x1, y1, x2, y2),
                        score = score,
                        classIndex = classIdx.toInt()
                    )
                )
            }
        }
        return detections
    }
}
