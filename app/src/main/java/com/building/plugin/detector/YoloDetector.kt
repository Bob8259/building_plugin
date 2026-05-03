package com.building.plugin.detector

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.RectF
import android.util.Log
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import androidx.core.graphics.createBitmap

data class DetectionResult(
    val boundingBox: RectF,
    val score: Float,
    val classIndex: Int
)

object YoloDetector {
    private const val TAG = "YoloDetector"
    private const val DEFAULT_MODEL_PATH = "obstacles_detector.tflite"

    private var appContext: Context? = null
    private var interpreter: Interpreter? = null
    private var currentModelType: String? = null
    private var inputWidth = 0
    private var inputHeight = 0

    // Dynamic output shape read from model
    private var outputNumDetections = 0
    private var outputDetectionSize = 0

    private var inputDataType: DataType = DataType.FLOAT32
    private var inputScale = 0f
    private var inputZeroPoint = 0

    fun initialize(context: Context) {
        appContext = context.applicationContext
    }

    fun isModelLoaded(): Boolean = interpreter != null

    fun getModelType(): String? = currentModelType

    /**
     * Loads model weights. Skips if the same model type is already loaded.
     * Throws on failure so the caller can report the error.
     */
    fun loadWeights(modelType: String) {
        if (interpreter != null && currentModelType == modelType) return

        clearWeights()

        val context = appContext
            ?: throw IllegalStateException("YoloDetector must be initialized before loading weights")

        val model = loadModelFile(context, modelType)
        val options = Interpreter.Options()
        interpreter = Interpreter(model, options)

        val inputTensor = interpreter!!.getInputTensor(0)
        val inputShape = inputTensor.shape() // [1, height, width, 3]
        inputHeight = inputShape[1]
        inputWidth = inputShape[2]
        inputDataType = inputTensor.dataType()

        // Read quantization params for quantized models
        if (inputDataType == DataType.INT8 || inputDataType == DataType.UINT8) {
            val quantization = inputTensor.quantizationParams()
            inputScale = quantization.scale
            inputZeroPoint = quantization.zeroPoint
        }

        // Read output tensor shape dynamically — e.g. [1, N, 6]
        val outputTensor = interpreter!!.getOutputTensor(0)
        val outputShape = outputTensor.shape()
        outputNumDetections = outputShape[1]
        outputDetectionSize = outputShape[2]

        currentModelType = modelType
        Log.i(TAG, "Model loaded: type=$modelType, input=${inputWidth}x${inputHeight}, output=[1, $outputNumDetections, $outputDetectionSize]")
    }

    fun clearWeights() {
        interpreter?.close()
        interpreter = null
        currentModelType = null
    }

    private fun loadModelFile(context: Context, modelType: String): MappedByteBuffer {
        // All YOLO models are expected to be bundled under app/src/main/assets.
        val assetFileName = when (modelType) {
            "walls-detect" -> "walls_detector.tflite"
            "numbers" -> "numbers_detector.tflite"
            "building-detect" -> "my_building_detector.tflite"
            // Keep the API modelType stable while loading the bundled capital detector asset.
            "capital-building-detect" -> "capital_building_detector.tflite"
            "remove-obstacle" -> DEFAULT_MODEL_PATH
            "clan-war-numbers" -> "clan_war_number_detector.tflite"
            "clan-game" -> "clan_game_detector.tflite"
            else -> throw IllegalArgumentException(
                "Unknown modelType: \"$modelType\". Valid types: walls-detect, numbers, building-detect, capital-building-detect, remove-obstacle, clan-war-numbers, clan-game"
            )
        }

        try {
            val fileDescriptor = context.assets.openFd(assetFileName)
            val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
            val fileChannel = inputStream.channel
            return fileChannel.map(
                FileChannel.MapMode.READ_ONLY,
                fileDescriptor.startOffset,
                fileDescriptor.declaredLength
            )
        } catch (e: IOException) {
            throw IllegalStateException(
                "Model asset \"$assetFileName\" for modelType \"$modelType\" was not found in app assets. " +
                    "Bundle it under app/src/main/assets and keep .tflite files uncompressed.",
                e
            )
        }
    }

    /**
     * Resizes bitmap to model input size with letterbox padding (black fill),
     * preserving aspect ratio. Returns the padded bitmap, scale factor, and offsets.
     */
    private fun resizeWithPadding(
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
     * Runs inference on the given bitmap. Model must already be loaded via loadWeights().
     * YOLO26 output does not require NMS — results are used directly.
     */
    fun detect(
        bitmap: Bitmap,
        clearWeightsAfter: Boolean = false,
        threshold: Float = 0.3f
    ): List<DetectionResult> {
        if (interpreter == null) return emptyList()

        var scaledBitmap: Bitmap? = null
        try {
            val (resized, scale, offset) = resizeWithPadding(bitmap, inputWidth, inputHeight)
            scaledBitmap = resized
            val (offX, offY) = offset

            val byteBuffer = convertBitmapToByteBuffer(scaledBitmap)

            // Allocate output buffer using dynamic shape [1, N, detectionSize]
            val output = Array(1) { Array(outputNumDetections) { FloatArray(outputDetectionSize) } }

            interpreter!!.run(byteBuffer, output)

            val detections = mutableListOf<DetectionResult>()
            for (detection in output[0]) {
                // detection: [x1, y1, x2, y2, score, class]
                val score = detection[4]
                if (score > threshold) {
                    // Map normalized coords back to original image space
                    val x1 = (detection[0] * inputWidth - offX) / scale
                    val y1 = (detection[1] * inputHeight - offY) / scale
                    val x2 = (detection[2] * inputWidth - offX) / scale
                    val y2 = (detection[3] * inputHeight - offY) / scale
                    val classIdx = detection[5]

                    detections.add(
                        DetectionResult(
                            boundingBox = RectF(x1, y1, x2, y2),
                            score = score,
                            classIndex = classIdx.toInt()
                        )
                    )
                }
            }
            return detections
        } finally {
            if (scaledBitmap != null && scaledBitmap != bitmap) {
                scaledBitmap.recycle()
            }
            if (clearWeightsAfter) {
                clearWeights()
            }
        }
    }

    private fun convertBitmapToByteBuffer(bitmap: Bitmap): ByteBuffer {
        val bufferSize = if (inputDataType == DataType.FLOAT32) {
            4 * inputWidth * inputHeight * 3
        } else {
            inputWidth * inputHeight * 3
        }

        val byteBuffer = ByteBuffer.allocateDirect(bufferSize)
        byteBuffer.order(ByteOrder.nativeOrder())

        // HARDWARE bitmaps (API 26+) don't support getPixels; copy to software config
        val softwareBitmap = if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.O &&
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
     * Filters out detections whose centers are closer than distanceThreshold,
     * keeping the higher-confidence one.
     */
    fun filterCloseDetections(
        detections: List<DetectionResult>,
        distanceThreshold: Double = 5.0
    ): List<DetectionResult> {
        val filtered = mutableListOf<DetectionResult>()

        for (detection in detections) {
            var isTooClose = false
            val iterator = filtered.listIterator()

            while (iterator.hasNext()) {
                val existing = iterator.next()
                val dx = detection.boundingBox.centerX() - existing.boundingBox.centerX()
                val dy = detection.boundingBox.centerY() - existing.boundingBox.centerY()
                val distance = kotlin.math.sqrt((dx * dx + dy * dy).toDouble())

                if (distance < distanceThreshold) {
                    isTooClose = true
                    if (detection.score > existing.score) {
                        iterator.remove()
                        iterator.add(detection)
                    }
                    break
                }
            }

            if (!isTooClose) {
                filtered.add(detection)
            }
        }
        return filtered
    }
}
