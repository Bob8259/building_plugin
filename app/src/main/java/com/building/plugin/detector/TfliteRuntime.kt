package com.building.plugin.detector

import android.content.Context
import android.graphics.Bitmap
import android.os.Build
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.io.IOException
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/**
 * Encapsulates TensorFlow Lite model loading and inference.
 */
internal class TfliteRuntime {

    private var interpreter: Interpreter? = null
    private var inputDataType: DataType = DataType.FLOAT32
    private var inputScale = 0f
    private var inputZeroPoint = 0

    var inputWidth = 0
        private set
    var inputHeight = 0
        private set
    var outputNumDetections = 0
        private set
    var outputDetectionSize = 0
        private set

    val isLoaded: Boolean get() = interpreter != null

    fun load(context: Context, modelType: String) {
        val model = loadModelFile(context, modelType)
        val options = Interpreter.Options()
        // Disable XNNPACK on devices below Android 9 (API 28) to prevent
        // native SIGSEGV crashes on older ARM CPUs (e.g. Cortex-A53).
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.P) {
            options.setUseXNNPACK(false)
        }
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
    }

    fun close() {
        interpreter?.close()
        interpreter = null
    }

    fun detect(
        scaledBitmap: Bitmap,
        scale: Float,
        offX: Float,
        offY: Float,
        threshold: Float
    ): List<DetectionResult> {
        val byteBuffer = ImagePreprocessor.convertBitmapToByteBuffer(
            scaledBitmap, inputWidth, inputHeight, inputDataType, inputScale, inputZeroPoint
        )

        // Allocate output buffer using dynamic shape [1, N, detectionSize]
        val output = Array(1) { Array(outputNumDetections) { FloatArray(outputDetectionSize) } }

        interpreter!!.run(byteBuffer, output)

        return ImagePreprocessor.parseDetections(output[0], inputWidth, inputHeight, scale, offX, offY, threshold)
    }

    companion object {
        private const val DEFAULT_MODEL_PATH = "obstacles_detector.tflite"

        private fun loadModelFile(context: Context, modelType: String): MappedByteBuffer {
            val assetFileName = when (modelType) {
                "walls-detect" -> "walls_detector.tflite"
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
    }
}
