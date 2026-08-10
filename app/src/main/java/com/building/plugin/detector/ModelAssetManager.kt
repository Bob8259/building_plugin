package com.building.plugin.detector

import android.content.Context
import android.util.Log
import java.io.File
import java.io.FileOutputStream
import java.io.IOException

/**
 * Manages extraction of ONNX model files from app assets into private storage.
 *
 * - On app start, all `.onnx` assets are extracted into `filesDir/models`,
 *   overwriting any existing copies (async, on a background thread).
 * - When a model is requested for loading, [resolveModelFile] returns the
 *   private-storage file. If it is missing, all models are re-extracted once
 *   (synchronously); if the file is still missing, an error is thrown.
 */
internal object ModelAssetManager {

    private const val TAG = "ModelAssetManager"
    private const val MODELS_DIR_NAME = "models"
    private const val ONNX_SUFFIX = ".onnx"
    private const val TMP_SUFFIX = ".tmp"

    /** Guards all extraction work so startup and fallback extraction never interleave. */
    private val extractionLock = Any()

    @Volatile
    private var extractionThread: Thread? = null

    /**
     * Starts extracting all `.onnx` assets into private storage in the background.
     * Existing files are always overwritten. Safe to call multiple times;
     * concurrent invocations are serialized by [extractionLock].
     */
    fun startExtraction(context: Context) {
        val appContext = context.applicationContext
        synchronized(extractionLock) {
            val existing = extractionThread
            if (existing != null && existing.isAlive) {
                Log.i(TAG, "Startup extraction already running, skipping duplicate trigger")
                return
            }
            val thread = Thread({
                try {
                    extractAllModels(appContext)
                } catch (e: Exception) {
                    Log.e(TAG, "Startup extraction failed", e)
                }
            }, "onnx-model-extract")
            thread.isDaemon = true
            extractionThread = thread
            thread.start()
        }
    }

    /**
     * Resolves the private-storage file for the given model asset name.
     *
     * 1. Returns the file immediately if it already exists and is non-empty.
     * 2. Otherwise re-extracts all `.onnx` assets synchronously (fallback).
     * 3. If the file is still missing, throws [IllegalStateException] with a
     *    descriptive error message so callers (e.g. the HTTP server) can
     *    report it to the user.
     */
    fun resolveModelFile(context: Context, assetName: String): File {
        val modelsDir = modelsDir(context)
        val target = File(modelsDir, assetName)

        if (isValidModelFile(target)) {
            return target
        }

        Log.w(TAG, "Model file \"$assetName\" missing in private storage, re-extracting all models")
        try {
            extractAllModels(context)
        } catch (e: Exception) {
            Log.e(TAG, "Fallback re-extraction failed", e)
        }

        if (isValidModelFile(target)) {
            return target
        }

        throw IllegalStateException(
            "Model file \"$assetName\" not found in private storage and re-extraction from assets failed. " +
                "Ensure the file is bundled under app/src/main/assets."
        )
    }

    private fun modelsDir(context: Context): File = File(context.filesDir, MODELS_DIR_NAME)

    private fun isValidModelFile(file: File): Boolean = file.exists() && file.isFile && file.length() > 0

    /**
     * Extracts every `.onnx` asset into [modelsDir], overwriting existing files.
     * Writes to a `.tmp` file first and renames atomically so readers never
     * observe a partially written model. Also removes stale files in the models
     * directory that no longer exist in assets.
     */
    private fun extractAllModels(context: Context) {
        synchronized(extractionLock) {
            val modelsDir = modelsDir(context)
            if (!modelsDir.exists() && !modelsDir.mkdirs()) {
                throw IOException("Failed to create models directory: ${modelsDir.absolutePath}")
            }

            val assetNames = listOnnxAssets(context)
            Log.i(TAG, "Extracting ${assetNames.size} ONNX model(s) to ${modelsDir.absolutePath}")

            for (assetName in assetNames) {
                extractSingle(context, modelsDir, assetName)
            }

            // Remove stale files that are no longer bundled in assets
            modelsDir.listFiles()?.forEach { file ->
                if (file.name.endsWith(ONNX_SUFFIX) && assetNames.none { it == file.name }) {
                    Log.i(TAG, "Removing stale model file: ${file.name}")
                    file.delete()
                }
            }

            Log.i(TAG, "Model extraction complete")
        }
    }

    /** Lists all `.onnx` files at the root of the assets directory. */
    private fun listOnnxAssets(context: Context): List<String> {
        return context.assets.list("")
            ?.filter { it.endsWith(ONNX_SUFFIX) }
            ?.sorted()
            ?: emptyList()
    }

    /** Copies one asset into the models directory using tmp-file + atomic rename. */
    private fun extractSingle(context: Context, modelsDir: File, assetName: String) {
        val target = File(modelsDir, assetName)
        val tmpFile = File(modelsDir, assetName + TMP_SUFFIX)

        try {
            context.assets.open(assetName).use { input ->
                FileOutputStream(tmpFile).use { output ->
                    input.copyTo(output)
                    output.flush()
                }
            }

            // Overwrite any existing file, then rename atomically
            if (target.exists()) {
                target.delete()
            }
            if (!tmpFile.renameTo(target)) {
                throw IOException("Failed to rename ${tmpFile.name} to ${target.name}")
            }
            Log.i(TAG, "Extracted $assetName (${target.length()} bytes)")
        } catch (e: IOException) {
            tmpFile.delete()
            throw IOException("Failed to extract model asset \"$assetName\" to private storage", e)
        }
    }
}
