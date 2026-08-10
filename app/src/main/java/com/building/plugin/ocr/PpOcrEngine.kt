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
import kotlin.math.ceil
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * PP-OCRv6 (small) text recognition engine: DB text detection + CTC recognition.
 *
 * Self-contained pipeline ported from the reference ONNX demo:
 *  1. det preprocess — scale long side to [limit], round dims to multiples of 32,
 *     normalize with ImageNet mean/std in BGR channel order.
 *  2. DB postprocess — binarize probability map, extract connected components,
 *     filter by mean-probability confidence, expand boxes (unclip), scale back.
 *  3. rec preprocess — crop each text box, resize to height 48, zero-pad width
 *     to at least 64, normalize to [-1, 1].
 *  4. CTC greedy decode — blank = index 0, space = dict size + 1.
 *
 * Models are loaded lazily on first [recognize] call (or eagerly via [load])
 * and stay resident until [clear] is called (explicitly or at service
 * shutdown). The OrtEnvironment is a process-wide singleton shared with the
 * detector, so [clear] only closes this engine's sessions and never the
 * environment itself.
 */
object PpOcrEngine {

    private const val TAG = "PpOcrEngine"

    private const val DET_ASSET = "ppocrv6_small_det.onnx"
    private const val REC_ASSET = "ppocrv6_small_rec.onnx"
    private const val DICT_ASSET = "ppocrv6_dict.json"

    /** rec fixed input height */
    private const val REC_HEIGHT = 48

    /** rec maximum resized width */
    private const val REC_MAX_WIDTH = 3200

    /** Default long-side limit for the detection input. */
    const val DEFAULT_LIMIT = 736

    /** Default minimum mean-probability confidence for a text box. */
    const val DEFAULT_BOX_THRESHOLD = 0.45f

    /** Default det probability binarization threshold. */
    const val DEFAULT_DET_THRESHOLD = 0.2f

    /** Default det box expansion (unclip) ratio: d = w * h * ratio / (2 * (w + h)). */
    const val DEFAULT_UNCLIP_RATIO = 1.4f

    /** Default rec minimum padded width (short labels -> faster). */
    const val DEFAULT_REC_MIN_WIDTH = 64

    private var appContext: Context? = null

    private var ortEnvironment: OrtEnvironment? = null
    private var detSession: OrtSession? = null
    private var recSession: OrtSession? = null
    private var dictChars: List<String>? = null

    /** Whether both det and rec sessions are currently loaded. */
    val isLoaded: Boolean
        @Synchronized get() = detSession != null && recSession != null && dictChars != null

    /** Stores the application context. Must be called before [recognize]. */
    fun initialize(context: Context) {
        appContext = context.applicationContext
    }

    /**
     * Pre-loads the det + rec models and dictionary into memory so the first
     * [recognize] call does not pay the model-loading latency.
     * No-op if already loaded. Throws on failure so callers can report it.
     */
    @Synchronized
    fun load() {
        if (isLoaded) return
        val context = appContext
            ?: throw IllegalStateException("PpOcrEngine must be initialized before loading")
        ensureLoaded(context)
    }

    /**
     * Closes the det/rec sessions and releases the dictionary.
     * Does NOT close the shared [OrtEnvironment] singleton.
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
     * Runs full OCR (detection + recognition) on the given bitmap.
     *
     * @param bitmap input image (not modified or recycled by this method)
     * @param limit long-side limit for the detection input (resized to multiples of 32)
     * @param boxThreshold minimum mean-probability confidence for a text box
     * @param detThreshold DB probability binarization threshold for the text mask
     * @param unclipRatio box expansion ratio: d = w * h * ratio / (2 * (w + h))
     * @param recMinWidth minimum padded width for the recognition input
     * @return recognized text lines sorted top-to-bottom, then left-to-right
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
            // Recover from a stale environment (e.g. env closed by the detector's
            // /clear) by reloading once, then retry. Rethrow on second failure.
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
            // 1. detect text lines
            val detInput = detPreprocess(src, limit)
            val prob = runDet(detInput)
            val boxes = dbPostprocess(
                prob, detInput.ratio, src.width, src.height,
                boxThreshold, detThreshold, unclipRatio
            )
                .sortedWith(compareBy({ it.y0 }, { it.x0 }))
            if (boxes.isEmpty()) {
                return emptyList()
            }

            // 2. recognize each line
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
                    val text = ctcDecode(logits, dict)
                    if (text.isNotEmpty()) {
                        results.add(
                            OcrResult(
                                text = text,
                                score = box.score,
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

    private fun ensureLoaded(context: Context) {
        if (isLoaded) return
        synchronized(this) {
            if (isLoaded) return

            val env = OrtEnvironment.getEnvironment()
            val detFile = ModelAssetManager.resolveModelFile(context, DET_ASSET)
            val recFile = ModelAssetManager.resolveModelFile(context, REC_ASSET)
            val dict = loadDict(context)

            detSession = env.createSession(detFile.absolutePath)
            recSession = env.createSession(recFile.absolutePath)
            ortEnvironment = env
            dictChars = dict
            Log.i(TAG, "PP-OCRv6 small loaded (det + rec), dict size=${dict.size}")
        }
    }

    /** Parses the character dictionary JSON array from assets. */
    private fun loadDict(context: Context): List<String> {
        val json = context.assets.open(DICT_ASSET).bufferedReader().use { it.readText() }
        val arr = JSONArray(json)
        val list = ArrayList<String>(arr.length())
        for (i in 0 until arr.length()) {
            list.add(arr.getString(i))
        }
        return list
    }

    // ── det: preprocess + inference + DB postprocess ────────────────────────

    /**
     * Scales the long side to [limit] (dims rounded to multiples of 32),
     * normalizes with ImageNet mean/std in BGR channel order, outputs NCHW.
     */
    private fun detPreprocess(bitmap: Bitmap, limit: Int): DetInput {
        val h = bitmap.height
        val w = bitmap.width
        val ratio = limit.toFloat() / max(h, w)
        val rh = max(((h * ratio) / 32f).roundToInt() * 32, 32)
        val rw = max(((w * ratio) / 32f).roundToInt() * 32, 32)

        val scaled = Bitmap.createScaledBitmap(bitmap, rw, rh, true)
        try {
            val pixels = IntArray(rw * rh)
            scaled.getPixels(pixels, 0, rw, 0, 0, rw, rh)

            val plane = rw * rh
            val data = FloatArray(3 * plane)
            for (i in pixels.indices) {
                val v = pixels[i]
                val b = (v and 0xFF) / 255f
                val g = (v shr 8 and 0xFF) / 255f
                val r = (v shr 16 and 0xFF) / 255f
                // BGR-order ImageNet normalization
                data[i] = (b - 0.485f) / 0.229f
                data[plane + i] = (g - 0.456f) / 0.224f
                data[2 * plane + i] = (r - 0.406f) / 0.225f
            }
            return DetInput(data, rw, rh, ratio)
        } finally {
            if (scaled !== bitmap) {
                scaled.recycle()
            }
        }
    }

    /** Runs the det model; returns the probability map as [ph][pw] rows. */
    private fun runDet(input: DetInput): Array<FloatArray> {
        val env = ortEnvironment ?: throw IllegalStateException("det session not loaded")
        val session = detSession ?: throw IllegalStateException("det session not loaded")

        val shape = longArrayOf(1, 3, input.height.toLong(), input.width.toLong())
        val tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(input.data), shape)
        tensor.use {
            val results = session.run(mapOf(session.inputNames.first() to it))
            results.use { r ->
                @Suppress("UNCHECKED_CAST")
                val out = r[0].value as Array<Array<Array<FloatArray>>> // [1, 1, ph, pw]
                return out[0][0]
            }
        }
    }

    /**
     * DB postprocess: binarize at [DET_THRESH], find 8-connected components,
     * drop components smaller than 3 px, filter by mean-probability score,
     * expand boxes by the unclip ratio, and scale back to original image coords.
     */
    private fun dbPostprocess(
        prob: Array<FloatArray>,
        ratio: Float,
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

            // unclip: expand the rect by area * ratio / perimeter
            val wBox = maxX - minX + 1
            val hBox = maxY - minY + 1
            val d = wBox * hBox * unclipRatio / (2f * (wBox + hBox))
            val x0 = max(0, (minX - d).toInt())
            val x1 = min(pw - 1, (maxX + d).toInt())
            val y0 = max(0, (minY - d).toInt())
            val y1 = min(ph - 1, (maxY + d).toInt())

            // scale back to original image coordinates
            val bx0 = max(0, (x0 / ratio).toInt())
            val by0 = max(0, (y0 / ratio).toInt())
            val bx1 = min(origW - 1, (x1 / ratio).toInt())
            val by1 = min(origH - 1, (y1 / ratio).toInt())

            if (bx1 > bx0 && by1 > by0) {
                boxes.add(TextBox(bx0, by0, bx1, by1, score))
            }
        }
        return boxes
    }

    // ── rec: preprocess + inference + CTC decode ────────────────────────────

    /**
     * Resizes the crop to height 48 (width clamped to 2..3200), zero-pads the
     * width to at least [minWidth], normalizes to [-1, 1], outputs NCHW.
     */
    private fun recPreprocess(crop: Bitmap, minWidth: Int): RecInput {
        val h = crop.height
        val w = crop.width
        val rw = ceil(REC_HEIGHT.toFloat() * w / h).toInt().coerceIn(2, REC_MAX_WIDTH)
        val iw = max(minWidth, rw)

        val scaled = Bitmap.createScaledBitmap(crop, rw, REC_HEIGHT, true)
        try {
            val pixels = IntArray(rw * REC_HEIGHT)
            scaled.getPixels(pixels, 0, rw, 0, 0, rw, REC_HEIGHT)

            val plane = REC_HEIGHT * iw
            val data = FloatArray(3 * plane) // zero-padded right side
            for (i in pixels.indices) {
                val v = pixels[i]
                val b = (v and 0xFF) / 255f
                val g = (v shr 8 and 0xFF) / 255f
                val r = (v shr 16 and 0xFF) / 255f
                data[i] = (b - 0.5f) / 0.5f
                data[plane + i] = (g - 0.5f) / 0.5f
                data[2 * plane + i] = (r - 0.5f) / 0.5f
            }
            return RecInput(data, iw)
        } finally {
            if (scaled !== crop) {
                scaled.recycle()
            }
        }
    }

    /** Runs the rec model; returns logits as [timeSteps][numClasses] rows. */
    private fun runRec(input: RecInput): Array<FloatArray> {
        val env = ortEnvironment ?: throw IllegalStateException("rec session not loaded")
        val session = recSession ?: throw IllegalStateException("rec session not loaded")

        val shape = longArrayOf(1, 3, REC_HEIGHT.toLong(), input.width.toLong())
        val tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(input.data), shape)
        tensor.use {
            val results = session.run(mapOf(session.inputNames.first() to it))
            results.use { r ->
                @Suppress("UNCHECKED_CAST")
                val out = r[0].value as Array<Array<FloatArray>> // [1, T, C]
                return out[0]
            }
        }
    }

    /**
     * CTC greedy decode: index 0 = blank, dict size + 1 = space,
     * 1..dict size = dictionary characters. Consecutive duplicates collapse.
     */
    private fun ctcDecode(logits: Array<FloatArray>, dict: List<String>): String {
        val sb = StringBuilder()
        var prev = -1
        for (row in logits) {
            var best = 0
            var bestValue = row[0]
            for (c in 1 until row.size) {
                if (row[c] > bestValue) {
                    bestValue = row[c]
                    best = c
                }
            }
            if (best == 0) {
                prev = -1
                continue
            }
            if (best == prev) continue
            prev = best
            when {
                best == dict.size + 1 -> sb.append(' ')
                best in 1..dict.size -> sb.append(dict[best - 1])
            }
        }
        return sb.toString()
    }

    // ── Helpers ─────────────────────────────────────────────────────────────

    /** Copies HARDWARE bitmaps into a software config so pixels can be read. */
    private fun toSoftwareBitmap(bitmap: Bitmap): Bitmap {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O &&
            bitmap.config == Bitmap.Config.HARDWARE
        ) {
            bitmap.copy(Bitmap.Config.ARGB_8888, false)
        } else {
            bitmap
        }
    }

    /** det input tensor data with its resized dims and scale ratio. */
    private data class DetInput(
        val data: FloatArray,
        val width: Int,
        val height: Int,
        val ratio: Float
    )

    /** rec input tensor data with its padded width. */
    private data class RecInput(
        val data: FloatArray,
        val width: Int
    )

    /** A detected text box in original image coordinates (inclusive bounds). */
    private data class TextBox(
        val x0: Int,
        val y0: Int,
        val x1: Int,
        val y1: Int,
        val score: Float
    )
}
