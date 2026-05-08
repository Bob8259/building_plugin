package com.building.plugin.detector

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.os.Build
import androidx.core.graphics.createBitmap
import org.tensorflow.lite.DataType
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Shared image preprocessing utilities used by both TFLite and ONNX runtimes.
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
     * Converts bitmap to ByteBuffer in NHWC format for TFLite.
     */
    fun convertBitmapToByteBuffer(
        bitmap: Bitmap,
        inputWidth: Int,
        inputHeight: Int,
        inputDataType: DataType,
        inputScale: Float,
        inputZeroPoint: Int
    ): ByteBuffer {
        val bufferSize = if (inputDataType == DataType.FLOAT32) {
            4 * inputWidth * inputHeight * 3
        } else {
            inputWidth * inputHeight * 3
        }

        val byteBuffer = ByteBuffer.allocateDirect(bufferSize)
        byteBuffer.order(ByteOrder.nativeOrder())

        // HARDWARE bitmaps (API 26+) don't support getPixels; copy to software config
        val softwareBitmap = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O &&
            bitmap.config == Bitmap.Config.HARDWARE
        ) {
            bitmap.copy(Bitmap.Config.ARGB_8888, false)
        } else {
            bitmap
        }

        val intValues = IntArray(inputWidth * inputHeight)
        softwareBitmap.getPixels(intValues, 0, softwareBitmap.width, 0, 0, softwareBitmap.width, softwareBitmap.height)

        var pixel = 0
        repeat(inputHeight) {
            repeat(inputWidth) {
                val value = intValues[pixel++]
                val r = (value shr 16 and 0xFF)
                val g = (value shr 8 and 0xFF)
                val b = (value and 0xFF)

                when (inputDataType) {
                    DataType.FLOAT32 -> {
                        byteBuffer.putFloat(r / 255.0f)
                        byteBuffer.putFloat(g / 255.0f)
                        byteBuffer.putFloat(b / 255.0f)
                    }
                    DataType.INT8 -> {
                        byteBuffer.put((r / 255.0f / inputScale + inputZeroPoint).toInt().toByte())
                        byteBuffer.put((g / 255.0f / inputScale + inputZeroPoint).toInt().toByte())
                        byteBuffer.put((b / 255.0f / inputScale + inputZeroPoint).toInt().toByte())
                    }
                    DataType.UINT8 -> {
                        byteBuffer.put(r.toByte())
                        byteBuffer.put(g.toByte())
                        byteBuffer.put(b.toByte())
                    }
                    else -> {}
                }
            }
        }

        if (softwareBitmap != bitmap) {
            softwareBitmap.recycle()
        }
        return byteBuffer
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
        threshold: Float
    ): List<DetectionResult> {
        val detections = mutableListOf<DetectionResult>()
        for (detection in output) {
            // detection: [x1, y1, x2, y2, score, class]
            val score = detection[4]
            if (score > threshold) {
                val x1 = (detection[0] * inputWidth - offX) / scale
                val y1 = (detection[1] * inputHeight - offY) / scale
                val x2 = (detection[2] * inputWidth - offX) / scale
                val y2 = (detection[3] * inputHeight - offY) / scale
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
