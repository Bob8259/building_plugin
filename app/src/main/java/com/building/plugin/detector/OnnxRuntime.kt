package com.building.plugin.detector

import android.graphics.Bitmap
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import java.io.File
import java.nio.FloatBuffer

/**
 * Encapsulates ONNX Runtime model loading and inference.
 */
internal class OnnxRuntime {

    private var ortEnvironment: OrtEnvironment? = null
    private var ortSession: OrtSession? = null

    var inputWidth = 0
        private set
    var inputHeight = 0
        private set
    var outputNumDetections = 0
        private set
    var outputDetectionSize = 0
        private set

    val isLoaded: Boolean get() = ortSession != null

    fun load(modelFile: File) {
        if (!modelFile.exists() || !modelFile.isFile) {
            throw IllegalStateException(
                "ONNX model file \"${modelFile.absolutePath}\" does not exist in private storage."
            )
        }

        val env = OrtEnvironment.getEnvironment()
        ortEnvironment = env
        val session = env.createSession(modelFile.absolutePath)
        ortSession = session

        // Read input shape from session metadata — expected [1, 3, H, W] (NCHW) or [1, H, W, 3] (NHWC)
        val inputInfo = session.inputInfo.values.first()
        val inputNodeInfo = inputInfo.info as TensorInfo
        val shape = inputNodeInfo.shape // e.g. [1, 3, 640, 640] or [1, 640, 640, 3]
        if (shape[1] == 3L) {
            // NCHW format
            inputHeight = shape[2].toInt()
            inputWidth = shape[3].toInt()
        } else {
            // NHWC format
            inputHeight = shape[1].toInt()
            inputWidth = shape[2].toInt()
        }

        // Read output shape — expected [1, N, detectionSize]
        val outputInfo = session.outputInfo.values.first()
        val outputNodeInfo = outputInfo.info as TensorInfo
        val outShape = outputNodeInfo.shape
        outputNumDetections = outShape[1].toInt()
        outputDetectionSize = outShape[2].toInt()
    }

    fun close() {
        ortSession?.close()
        ortSession = null
        ortEnvironment?.close()
        ortEnvironment = null
    }

    fun detect(
        scaledBitmap: Bitmap,
        scale: Float,
        offX: Float,
        offY: Float,
        threshold: Float
    ): List<DetectionResult> {
        val env = ortEnvironment!!
        val session = ortSession!!

        // Prepare input as NCHW float array
        val floatArray = ImagePreprocessor.convertBitmapToFloatArrayNCHW(scaledBitmap, inputWidth, inputHeight)
        val inputShape = longArrayOf(1, 3, inputHeight.toLong(), inputWidth.toLong())
        val inputTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(floatArray), inputShape)

        val inputName = session.inputNames.first()
        val results = session.run(mapOf(inputName to inputTensor))

        // Output shape: [1, N, detectionSize]
        @Suppress("UNCHECKED_CAST")
        val outputArray = results[0].value as Array<Array<FloatArray>>

        val detections = ImagePreprocessor.parseDetections(
            outputArray[0], inputWidth, inputHeight, scale, offX, offY, threshold,
            normalizedCoords = false
        )

        results.close()
        inputTensor.close()

        return detections
    }
}
