package com.building.plugin.ocr

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import android.os.Build
import android.util.Log
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import com.building.plugin.detector.ModelAssetManager
import org.json.JSONArray
import java.nio.FloatBuffer
import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * PP-OCRv6 Small (FP16) text recognition engine.
 * Implemented in pure Kotlin + Android native graphics with zero OpenCV dependencies.
 *
 * Pipeline:
 *  1. det preprocess — scale long side to [limit], round dims to multiples of 32,
 *     normalize with ImageNet mean/std in BGR channel order.
 *  2. DB postprocess — binarize probability map, extract 8-connected components,
 *     filter by mean confidence, expand boxes (unclip), scale back to original image space,
 *     and sort into natural top-to-bottom reading order.
 *  3. rec preprocess — crop each text box, resize to height 48, zero-pad width
 *     to at least [recMinWidth], normalize to [-1, 1] in RGB channel order.
 *  4. CTC greedy decode — blank = index 0, dict characters = index 1..N, space = index N+1.
 */
object PpOcrEngine {

    private const val TAG = "PpOcrEngine"

    private const val DET_ASSET = "det_fp16.onnx"
    private const val REC_ASSET = "rec_fp16.onnx"
    private const val CHARACTERS_ASSET = "rec_characters.json"

    /** Fixed input height for recognizer */
    private const val REC_HEIGHT = 48

    /** Maximum resized width for recognizer */
    private const val REC_MAX_WIDTH = 3200

    /** Maximum side limit for detector resize */
    private const val DET_MAX_SIDE = 4000

    /** Default long-side limit for the detection input */
    const val DEFAULT_LIMIT = 960

    /** Default minimum mean-probability confidence for a text box */
    const val DEFAULT_BOX_THRESHOLD = 0.45f

    /** Default det probability binarization threshold */
    const val DEFAULT_DET_THRESHOLD = 0.20f

    /** Default det box expansion (unclip) ratio */
    const val DEFAULT_UNCLIP_RATIO = 1.4f

    /** Default rec minimum padded width */
    const val DEFAULT_REC_MIN_WIDTH = 320

    private var appContext: Context? = null
    private var ortEnvironment: OrtEnvironment? = null
    private var detSession: OrtSession? = null
    private var recSession: OrtSession? = null
    private var dictChars: List<String>? = null

    /** Whether det and rec sessions are currently loaded. */
    val isLoaded: Boolean
        @Synchronized get() = detSession != null && recSession != null && dictChars != null

    /** Whether rec session and dictionary are currently loaded. */
    val isRecLoaded: Boolean
        @Synchronized get() = recSession != null && dictChars != null

    /** Stores the application context. Must be called before [recognize] or [load]. */
    fun initialize(context: Context) {
        appContext = context.applicationContext
    }

    /**
     * Pre-loads the det + rec models and dictionary into memory.
     * No-op if already loaded.
     */
    @Synchronized
    fun load() {
        if (isLoaded) return
        val context = appContext
            ?: throw IllegalStateException("PpOcrEngine must be initialized before loading")
        ensureLoaded(context)
    }

    /**
     * Pre-loads only the recognition model and dictionary into memory.
     * No-op if already loaded.
     */
    @Synchronized
    fun loadRec() {
        if (isRecLoaded) return
        val context = appContext
            ?: throw IllegalStateException("PpOcrEngine must be initialized before loading")
        ensureRecLoaded(context)
    }

    /**
     * Closes the active det/rec sessions and releases model memory.
     */
    @Synchronized
    fun clear() {
        detSession?.close()
        recSession?.close()
        detSession = null
        recSession = null
        dictChars = null
        ortEnvironment = null
    }

    /**
     * Runs text recognition only (no detection model) on the given cropped bitmap.
     *
     * @param cropBitmap cropped image patch containing text
     * @param recMinWidth minimum width for recognition padding (default 320)
     * @return recognized text and recognition confidence
     */
    @Synchronized
    fun recognizeText(
        cropBitmap: Bitmap,
        recMinWidth: Int = DEFAULT_REC_MIN_WIDTH
    ): RecResult {
        return try {
            recognizeTextInternal(cropBitmap, recMinWidth)
        } catch (e: Exception) {
            Log.w(TAG, "OCR recognition inference failed, reloading rec model and retrying once", e)
            recSession?.close()
            recSession = null
            val context = appContext
                ?: throw IllegalStateException("PpOcrEngine must be initialized before use")
            ensureRecLoaded(context)
            recognizeTextInternal(cropBitmap, recMinWidth)
        }
    }

    /**
     * Runs text recognition only on the specified sub-rectangle [left, top, right, bottom]
     * of the given bitmap without running the detection model.
     */
    @Synchronized
    fun recognizeRegion(
        bitmap: Bitmap,
        left: Int,
        top: Int,
        right: Int,
        bottom: Int,
        recMinWidth: Int = DEFAULT_REC_MIN_WIDTH
    ): RecResult {
        val clampedL = left.coerceIn(0, bitmap.width - 1)
        val clampedT = top.coerceIn(0, bitmap.height - 1)
        val clampedR = right.coerceIn(clampedL + 1, bitmap.width)
        val clampedB = bottom.coerceIn(clampedT + 1, bitmap.height)
        val w = clampedR - clampedL
        val h = clampedB - clampedT

        val crop = Bitmap.createBitmap(bitmap, clampedL, clampedT, w, h)
        try {
            return recognizeText(crop, recMinWidth)
        } finally {
            crop.recycle()
        }
    }

    private fun recognizeTextInternal(
        cropBitmap: Bitmap,
        recMinWidth: Int
    ): RecResult {
        val context = appContext
            ?: throw IllegalStateException("PpOcrEngine must be initialized before use")
        ensureRecLoaded(context)

        val src = toSoftwareBitmap(cropBitmap)
        try {
            val dict = dictChars ?: throw IllegalStateException("Dictionary not loaded")
            val recInput = recPreprocess(src, recMinWidth)
            val logits = runRec(recInput)
            val (text, confidence) = ctcDecode(logits, dict)
            return RecResult(text = text, confidence = confidence)
        } finally {
            if (src !== cropBitmap) {
                src.recycle()
            }
        }
    }

    /**
     * Runs full OCR (detection + recognition) on the given bitmap.
     *
     * @param bitmap input image (not recycled by this method)
     * @param limit long-side limit for detection input
     * @param boxThreshold minimum confidence for a text box
     * @param detThreshold DB probability binarization threshold
     * @param unclipRatio box expansion ratio
     * @param recMinWidth minimum padded width for recognition input
     * @return recognized text lines sorted in natural reading order
     */
    @Synchronized
    fun recognize(
        bitmap: Bitmap,
        limit: Int = DEFAULT_LIMIT,
        boxThreshold: Float = DEFAULT_BOX_THRESHOLD,
        detThreshold: Float = DEFAULT_DET_THRESHOLD,
        unclipRatio: Float = DEFAULT_UNCLIP_RATIO,
        recMinWidth: Int = DEFAULT_REC_MIN_WIDTH
    ): List<OcrResult> {
        return try {
            recognizeInternal(bitmap, limit, boxThreshold, detThreshold, unclipRatio, recMinWidth)
        } catch (e: Exception) {
            Log.w(TAG, "OCR inference failed, reloading models and retrying once", e)
            clear()
            recognizeInternal(bitmap, limit, boxThreshold, detThreshold, unclipRatio, recMinWidth)
        }
    }

    // ── Pipeline ────────────────────────────────────────────────────────────

    private fun recognizeInternal(
        bitmap: Bitmap,
        limit: Int,
        boxThreshold: Float,
        detThreshold: Float,
        unclipRatio: Float,
        recMinWidth: Int
    ): List<OcrResult> {
        val context = appContext
            ?: throw IllegalStateException("PpOcrEngine must be initialized before use")
        ensureLoaded(context)

        val src = toSoftwareBitmap(bitmap)
        try {
            // 1. Detection
            val detInput = detPreprocess(src, limit)
            val prob = runDet(detInput)
            val rawBoxes = dbPostprocess(
                prob = prob,
                origW = src.width,
                origH = src.height,
                boxThreshold = boxThreshold,
                detThreshold = detThreshold,
                unclipRatio = unclipRatio
            )
            if (rawBoxes.isEmpty()) {
                return emptyList()
            }

            // Sort boxes into natural reading order (row by row, left to right)
            val boxes = sortBoxesIntoReadingOrder(rawBoxes)

            // 2. Recognition for each box
            val dict = dictChars ?: throw IllegalStateException("Dictionary not loaded")
            val results = mutableListOf<OcrResult>()
            for (box in boxes) {
                val cropW = box.x1 - box.x0 + 1
                val cropH = box.y1 - box.y0 + 1
                if (cropW <= 0 || cropH <= 0) continue

                val crop = Bitmap.createBitmap(src, box.x0, box.y0, cropW, cropH)
                try {
                    val recInput = recPreprocess(crop, recMinWidth)
                    val logits = runRec(recInput)
                    val (text, confidence) = ctcDecode(logits, dict)

                    if (text.isNotEmpty()) {
                        results.add(
                            OcrResult(
                                text = text,
                                score = box.score,
                                confidence = confidence,
                                box = RectF(
                                    box.x0.toFloat(),
                                    box.y0.toFloat(),
                                    box.x1.toFloat(),
                                    box.y1.toFloat()
                                )
                            )
                        )
                    }
                } finally {
                    crop.recycle()
                }
            }
            return results
        } finally {
            if (src !== bitmap) {
                src.recycle()
            }
        }
    }

    // ── Model loading ───────────────────────────────────────────────────────

    @Synchronized
    private fun ensureDetLoaded(context: Context) {
        if (detSession != null) return

        val env = ortEnvironment ?: OrtEnvironment.getEnvironment()
        val detFile = ModelAssetManager.resolveModelFile(context, DET_ASSET)
        val sessionOptions = OrtSession.SessionOptions().apply {
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        }
        detSession = env.createSession(detFile.absolutePath, sessionOptions)
        ortEnvironment = env
        Log.i(TAG, "PP-OCRv6 Small Det (FP16) loaded")
    }

    @Synchronized
    private fun ensureRecLoaded(context: Context) {
        if (recSession != null && dictChars != null) return

        val env = ortEnvironment ?: OrtEnvironment.getEnvironment()
        val recFile = ModelAssetManager.resolveModelFile(context, REC_ASSET)
        val dict = dictChars ?: loadDict(context)

        val sessionOptions = OrtSession.SessionOptions().apply {
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        }
        recSession = env.createSession(recFile.absolutePath, sessionOptions)
        ortEnvironment = env
        dictChars = dict
        Log.i(TAG, "PP-OCRv6 Small Rec (FP16) loaded, dict size=${dict.size}")
    }

    @Synchronized
    private fun ensureLoaded(context: Context) {
        if (isLoaded) return
        ensureDetLoaded(context)
        ensureRecLoaded(context)
    }

    /** Loads the character dictionary JSON array from assets. */
    private fun loadDict(context: Context): List<String> {
        val json = context.assets.open(CHARACTERS_ASSET).bufferedReader(Charsets.UTF_8).use { it.readText() }
        val arr = JSONArray(json)
        val list = ArrayList<String>(arr.length())
        for (i in 0 until arr.length()) {
            list.add(arr.getString(i))
        }
        return list
    }

    // ── Detection: Preprocess + Inference + Postprocess ───────────────────────

    private fun detPreprocess(bitmap: Bitmap, limit: Int): DetInput {
        val origW = bitmap.width
        val origH = bitmap.height

        val maxSide = max(origW, origH)
        val ratio = if (maxSide > limit) {
            limit.toFloat() / maxSide.toFloat()
        } else {
            1.0f
        }

        var rw = (origW * ratio).roundToInt()
        var rh = (origH * ratio).roundToInt()
        val maxResized = max(rw, rh)
        if (maxResized > DET_MAX_SIDE) {
            val r2 = DET_MAX_SIDE.toFloat() / maxResized.toFloat()
            rw = (rw * r2).roundToInt()
            rh = (rh * r2).roundToInt()
        }

        rw = max((rw / 32f).roundToInt() * 32, 32)
        rh = max((rh / 32f).roundToInt() * 32, 32)

        val scaled = Bitmap.createScaledBitmap(bitmap, rw, rh, true)
        try {
            val pixels = IntArray(rw * rh)
            scaled.getPixels(pixels, 0, rw, 0, 0, rw, rh)

            val plane = rw * rh
            val data = FloatArray(3 * plane)
            // Normalized with ImageNet mean/std in BGR channel order
            for (i in 0 until plane) {
                val pixel = pixels[i]
                val b = (pixel and 0xFF) / 255.0f
                val g = ((pixel shr 8) and 0xFF) / 255.0f
                val r = ((pixel shr 16) and 0xFF) / 255.0f

                data[i] = (b - 0.485f) / 0.229f
                data[plane + i] = (g - 0.456f) / 0.224f
                data[2 * plane + i] = (r - 0.406f) / 0.225f
            }
            return DetInput(data, rw, rh)
        } finally {
            if (scaled !== bitmap) {
                scaled.recycle()
            }
        }
    }

    private fun runDet(input: DetInput): Array<FloatArray> {
        val env = ortEnvironment ?: throw IllegalStateException("det session not loaded")
        val session = detSession ?: throw IllegalStateException("det session not loaded")

        val shape = longArrayOf(1, 3, input.height.toLong(), input.width.toLong())
        val tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(input.data), shape)
        tensor.use {
            val results = session.run(mapOf(session.inputNames.first() to it))
            results.use { r ->
                @Suppress("UNCHECKED_CAST")
                val out = r[0].value as Array<Array<Array<FloatArray>>>
                return out[0][0]
            }
        }
    }

    private fun dbPostprocess(
        prob: Array<FloatArray>,
        origW: Int,
        origH: Int,
        boxThreshold: Float,
        detThreshold: Float,
        unclipRatio: Float
    ): List<TextBox> {
        val ph = prob.size
        val pw = prob[0].size
        val total = ph * pw

        val mask = BooleanArray(total)
        for (y in 0 until ph) {
            val row = prob[y]
            val base = y * pw
            for (x in 0 until pw) {
                if (row[x] > detThreshold) mask[base + x] = true
            }
        }

        val visited = BooleanArray(total)
        val stack = IntArray(total)
        val boxes = mutableListOf<TextBox>()

        val scaleX = origW.toFloat() / pw.toFloat()
        val scaleY = origH.toFloat() / ph.toFloat()

        for (start in 0 until total) {
            if (!mask[start] || visited[start]) continue

            var top = 0
            stack[top++] = start
            visited[start] = true

            var count = 0
            var sum = 0f
            var minX = Int.MAX_VALUE
            var maxX = Int.MIN_VALUE
            var minY = Int.MAX_VALUE
            var maxY = Int.MIN_VALUE

            while (top > 0) {
                val idx = stack[--top]
                val x = idx % pw
                val y = idx / pw

                count++
                sum += prob[y][x]
                if (x < minX) minX = x
                if (x > maxX) maxX = x
                if (y < minY) minY = y
                if (y > maxY) maxY = y

                for (dy in -1..1) {
                    val ny = y + dy
                    if (ny < 0 || ny >= ph) continue
                    for (dx in -1..1) {
                        if (dx == 0 && dy == 0) continue
                        val nx = x + dx
                        if (nx < 0 || nx >= pw) continue
                        val nIdx = ny * pw + nx
                        if (mask[nIdx] && !visited[nIdx]) {
                            visited[nIdx] = true
                            stack[top++] = nIdx
                        }
                    }
                }
            }

            if (count < 3) continue
            val score = sum / count
            if (score < boxThreshold) continue

            val wBox = maxX - minX + 1
            val hBox = maxY - minY + 1
            if (min(wBox, hBox) < 3) continue

            // Unclip expansion
            val d = wBox * hBox * unclipRatio / (2f * (wBox + hBox))
            val x0 = max(0, (minX - d).toInt())
            val x1 = min(pw - 1, (maxX + d).toInt())
            val y0 = max(0, (minY - d).toInt())
            val y1 = min(ph - 1, (maxY + d).toInt())

            val finalW = x1 - x0 + 1
            val finalH = y1 - y0 + 1
            if (min(finalW, finalH) < 5) continue

            // Scale coordinates back to original image space
            val bx0 = (x0 * scaleX).roundToInt().coerceIn(0, origW - 1)
            val by0 = (y0 * scaleY).roundToInt().coerceIn(0, origH - 1)
            val bx1 = (x1 * scaleX).roundToInt().coerceIn(0, origW - 1)
            val by1 = (y1 * scaleY).roundToInt().coerceIn(0, origH - 1)

            if (bx1 > bx0 && by1 > by0) {
                boxes.add(TextBox(bx0, by0, bx1, by1, score))
            }
        }
        return boxes
    }

    private fun sortBoxesIntoReadingOrder(boxes: List<TextBox>): List<TextBox> {
        if (boxes.size <= 1) return boxes
        val sortedByY = boxes.sortedBy { (it.y0 + it.y1) / 2f }
        val rows = mutableListOf<MutableList<TextBox>>()

        for (box in sortedByY) {
            val centerY = (box.y0 + box.y1) / 2f
            val height = max(1f, (box.y1 - box.y0).toFloat())
            var matchingRow: MutableList<TextBox>? = null
            for (row in rows) {
                val rowCenter = row.map { (it.y0 + it.y1) / 2f }.average().toFloat()
                val rowHeight = row.map { (it.y1 - it.y0).toFloat() }.average().toFloat()
                if (abs(centerY - rowCenter) <= 0.5f * min(height, rowHeight)) {
                    matchingRow = row
                    break
                }
            }
            if (matchingRow == null) {
                rows.add(mutableListOf(box))
            } else {
                matchingRow.add(box)
            }
        }
        rows.sortBy { row -> row.map { (it.y0 + it.y1) / 2f }.average().toFloat() }
        return rows.flatMap { row -> row.sortedBy { it.x0 } }
    }

    // ── Recognition: Preprocess + Inference + CTC Decode ─────────────────────

    private fun recPreprocess(crop: Bitmap, minWidth: Int): RecInput {
        val h = crop.height
        val w = crop.width
        val rw = ceil(REC_HEIGHT.toFloat() * w / h).toInt().coerceIn(2, REC_MAX_WIDTH)
        val targetW = max(minWidth, rw)

        val scaled = Bitmap.createScaledBitmap(crop, rw, REC_HEIGHT, true)
        try {
            val pixels = IntArray(rw * REC_HEIGHT)
            scaled.getPixels(pixels, 0, rw, 0, 0, rw, REC_HEIGHT)

            val plane = REC_HEIGHT * targetW
            val data = FloatArray(3 * plane) // Zero-padded by default
            for (y in 0 until REC_HEIGHT) {
                val rowOffset = y * targetW
                val pixelRowOffset = y * rw
                for (x in 0 until rw) {
                    val pixel = pixels[pixelRowOffset + x]
                    val r = ((pixel shr 16) and 0xFF) / 255.0f
                    val g = ((pixel shr 8) and 0xFF) / 255.0f
                    val b = (pixel and 0xFF) / 255.0f

                    // RGB order normalized to [-1, 1]
                    data[rowOffset + x] = (r - 0.5f) / 0.5f
                    data[plane + rowOffset + x] = (g - 0.5f) / 0.5f
                    data[2 * plane + rowOffset + x] = (b - 0.5f) / 0.5f
                }
            }
            return RecInput(data, targetW)
        } finally {
            if (scaled !== crop) {
                scaled.recycle()
            }
        }
    }

    private fun runRec(input: RecInput): Array<FloatArray> {
        val env = ortEnvironment ?: throw IllegalStateException("rec session not loaded")
        val session = recSession ?: throw IllegalStateException("rec session not loaded")

        val shape = longArrayOf(1, 3, REC_HEIGHT.toLong(), input.width.toLong())
        val tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(input.data), shape)
        tensor.use {
            val results = session.run(mapOf(session.inputNames.first() to it))
            results.use { r ->
                @Suppress("UNCHECKED_CAST")
                val out = r[0].value as Array<Array<FloatArray>>
                return out[0]
            }
        }
    }

    private fun ctcDecode(logits: Array<FloatArray>, dict: List<String>): Pair<String, Float> {
        val sb = StringBuilder()
        val confidences = mutableListOf<Float>()
        var prev = -1
        val spaceIndex = dict.size + 1

        for (row in logits) {
            var best = 0
            var bestValue = row[0]
            for (c in 1 until row.size) {
                if (row[c] > bestValue) {
                    bestValue = row[c]
                    best = c
                }
            }

            if (best != 0 && best != prev) {
                when {
                    best in 1..dict.size -> {
                        sb.append(dict[best - 1])
                        confidences.add(bestValue)
                    }
                    best == spaceIndex -> {
                        sb.append(' ')
                        confidences.add(bestValue)
                    }
                }
            }
            prev = best
        }

        val text = sb.toString().trim()
        val confidence = if (confidences.isNotEmpty()) confidences.average().toFloat() else 0f
        return Pair(text, confidence)
    }

    // ── Helpers ─────────────────────────────────────────────────────────────

    private fun toSoftwareBitmap(bitmap: Bitmap): Bitmap {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O &&
            bitmap.config == Bitmap.Config.HARDWARE
        ) {
            bitmap.copy(Bitmap.Config.ARGB_8888, false)
        } else {
            bitmap
        }
    }

    private data class DetInput(
        val data: FloatArray,
        val width: Int,
        val height: Int
    )

    private data class RecInput(
        val data: FloatArray,
        val width: Int
    )

    private data class TextBox(
        val x0: Int,
        val y0: Int,
        val x1: Int,
        val y1: Int,
        val score: Float
    )
}
