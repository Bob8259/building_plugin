package com.building.plugin.detector

import android.content.Context
import android.graphics.Bitmap
import android.util.Log

private enum class ActiveRuntime { TFLITE, ONNX }

object BuildingDetector {
    private const val TAG = "BuildingDetector"

    private var appContext: Context? = null
    private var currentModelType: String? = null
    private var activeRuntime: ActiveRuntime? = null

    private var tfliteRuntime: TfliteRuntime? = null
    private var onnxRuntime: OnnxRuntime? = null

    fun initialize(context: Context) {
        appContext = context.applicationContext
    }

    fun isModelLoaded(): Boolean = tfliteRuntime?.isLoaded == true || onnxRuntime?.isLoaded == true

    fun getModelType(): String? = currentModelType

    /**
     * Loads model weights. Skips if the same model type is already loaded.
     * Throws on failure so the caller can report the error.
     */
    fun loadWeights(modelType: String) {
        if (isModelLoaded() && currentModelType == modelType) return

        clearWeights()

        val context = appContext
            ?: throw IllegalStateException("BuildingDetector must be initialized before loading weights")

        if (modelType == "building-detect" || modelType == "capital-building-detect" || modelType == "clan-game" || modelType == "clan-war-numbers" || modelType == "numbers" || modelType == "remove-obstacle") {
            val assetName = when (modelType) {
                "building-detect" -> "my_building_detector.onnx"
                "capital-building-detect" -> "capital_building_detector.onnx"
                "clan-game" -> "clan_game_detector.onnx"
                "clan-war-numbers" -> "clan_war_number_detector.onnx"
                "numbers" -> "numbers_detector.onnx"
                "remove-obstacle" -> "obstacles_detector.onnx"
                else -> error("unreachable")
            }
            val runtime = OnnxRuntime()
            runtime.load(context, assetName)
            onnxRuntime = runtime
            activeRuntime = ActiveRuntime.ONNX
        } else {
            val runtime = TfliteRuntime()
            runtime.load(context, modelType)
            tfliteRuntime = runtime
            activeRuntime = ActiveRuntime.TFLITE
        }

        currentModelType = modelType
        val (w, h) = inputDimensions()
        Log.i(TAG, "Model loaded: type=$modelType, runtime=$activeRuntime, input=${w}x${h}")
    }

    fun clearWeights() {
        tfliteRuntime?.close()
        tfliteRuntime = null

        onnxRuntime?.close()
        onnxRuntime = null

        activeRuntime = null
        currentModelType = null
    }

    /**
     * Runs inference on the given bitmap. Model must already be loaded via loadWeights().
     */
    fun detect(
        bitmap: Bitmap,
        clearWeightsAfter: Boolean = false,
        threshold: Float = 0.3f
    ): List<DetectionResult> {
        if (!isModelLoaded()) return emptyList()

        val (inputWidth, inputHeight) = inputDimensions()
        var scaledBitmap: Bitmap? = null
        try {
            val (resized, scale, offset) = ImagePreprocessor.resizeWithPadding(bitmap, inputWidth, inputHeight)
            scaledBitmap = resized
            val (offX, offY) = offset

            val detections = when (activeRuntime) {
                ActiveRuntime.ONNX -> onnxRuntime!!.detect(scaledBitmap, scale, offX, offY, threshold)
                ActiveRuntime.TFLITE -> tfliteRuntime!!.detect(scaledBitmap, scale, offX, offY, threshold)
                else -> emptyList()
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

    private fun inputDimensions(): Pair<Int, Int> = when (activeRuntime) {
        ActiveRuntime.ONNX -> Pair(onnxRuntime!!.inputWidth, onnxRuntime!!.inputHeight)
        ActiveRuntime.TFLITE -> Pair(tfliteRuntime!!.inputWidth, tfliteRuntime!!.inputHeight)
        else -> Pair(0, 0)
    }
}
