package com.building.plugin.detector

import android.content.Context
import android.graphics.Bitmap
import android.util.Log

object BuildingDetector {
    private const val TAG = "BuildingDetector"

    private var appContext: Context? = null
    private var currentModelType: String? = null
    private var onnxRuntime: OnnxRuntime? = null

    fun initialize(context: Context) {
        appContext = context.applicationContext
        // Extract all ONNX models from assets into private storage on app start,
        // overwriting any existing files.
        ModelAssetManager.startExtraction(context)
    }

    fun isModelLoaded(): Boolean = onnxRuntime?.isLoaded == true

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

        val assetName = when (modelType) {
            "building-detect" -> "my_building_detector.onnx"
            "capital-building-detect" -> "capital_building_detector.onnx"
            "clan-game" -> "clan_game_detector.onnx"
            "clan-war-numbers" -> "clan_war_number_detector.onnx"
            "main-base-battle" -> "main_base_battle.onnx"
            "numbers" -> "numbers_detector.onnx"
            "remove-obstacle" -> "obstacles_detector.onnx"
            "walls-detect" -> "walls_detector.onnx"
            else -> throw IllegalArgumentException(
                "Unknown modelType: \"$modelType\". Valid types: walls-detect, numbers, building-detect, capital-building-detect, remove-obstacle, clan-war-numbers, clan-game, main-base-battle"
            )
        }

        // Resolve the model file in private storage; falls back to re-extracting
        // all models once and throws a descriptive error if still missing.
        val modelFile = ModelAssetManager.resolveModelFile(context, assetName)

        val runtime = OnnxRuntime()
        runtime.load(modelFile)
        onnxRuntime = runtime

        currentModelType = modelType
        Log.i(TAG, "Model loaded: type=$modelType, input=${runtime.inputWidth}x${runtime.inputHeight}")
    }

    fun clearWeights() {
        onnxRuntime?.close()
        onnxRuntime = null
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
        val runtime = onnxRuntime ?: return emptyList()

        var scaledBitmap: Bitmap? = null
        try {
            val (resized, scale, offset) = ImagePreprocessor.resizeWithPadding(
                bitmap, runtime.inputWidth, runtime.inputHeight
            )
            scaledBitmap = resized
            val (offX, offY) = offset

            return runtime.detect(scaledBitmap, scale, offX, offY, threshold)
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
}
