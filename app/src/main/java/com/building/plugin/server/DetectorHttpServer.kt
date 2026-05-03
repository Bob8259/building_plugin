package com.building.plugin.server

import android.graphics.BitmapFactory
import android.util.Log
import fi.iki.elonen.NanoHTTPD
import org.json.JSONArray
import org.json.JSONObject
import com.building.plugin.detector.BuildingDetector

/**
 * Lightweight HTTP server exposing detector inference endpoints on the given port.
 *
 * Endpoints:
 *   GET  /status  — health check
 *   POST /load    — load model weights
 *   POST /detect  — run inference on a posted image
 *   POST /clear   — unload model weights
 */
class DetectorHttpServer(port: Int) : NanoHTTPD(port) {

    companion object {
        private const val TAG = "DetectorHttpServer"
    }

    override fun serve(session: IHTTPSession): Response {
        val uri = session.uri
        val method = session.method

        return try {
            when (method) {
                Method.GET if uri == "/status" -> handleStatus()
                Method.POST if uri == "/load" -> handleLoad(session)
                Method.POST if uri == "/detect" -> handleDetect(session)
                Method.POST if uri == "/clear" -> handleClear()
                else -> jsonResponse(
                    Response.Status.NOT_FOUND,
                    JSONObject().put("error", "Not found: $method $uri")
                )
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error handling $method $uri", e)
            jsonResponse(
                Response.Status.INTERNAL_ERROR,
                JSONObject().put("error", e.message ?: "Unknown error")
            )
        }
    }

    // ── GET /status ─────────────────────────────────────────────────────────
    private fun handleStatus(): Response {
        val json = JSONObject()
            .put("status", "running")
            .put("version", "1.08")
            .put("modelLoaded", BuildingDetector.isModelLoaded())
            .put("modelType", BuildingDetector.getModelType() ?: JSONObject.NULL)
        return jsonResponse(Response.Status.OK, json)
    }

    // ── POST /load ──────────────────────────────────────────────────────────
    private fun handleLoad(session: IHTTPSession): Response {
        val body = readBody(session)
        val json = JSONObject(body)

        // modelType is required — reject if missing or null
        if (!json.has("modelType") || json.isNull("modelType")) {
            return jsonResponse(
                Response.Status.BAD_REQUEST,
                JSONObject().put("success", false)
                    .put("error", "\"modelType\" is required. Valid types: walls-detect, numbers, building-detect, capital-building-detect, remove-obstacle, clan-war-numbers, clan-game")
            )
        }
        val modelType = json.getString("modelType")

        return try {
            BuildingDetector.loadWeights(modelType)
            jsonResponse(Response.Status.OK, JSONObject().put("success", true))
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load model", e)
            jsonResponse(
                Response.Status.INTERNAL_ERROR,
                JSONObject().put("success", false).put("error", e.message ?: "Unknown error")
            )
        }
    }

    // ── POST /detect ────────────────────────────────────────────────────────
    private fun handleDetect(session: IHTTPSession): Response {
        if (!BuildingDetector.isModelLoaded()) {
            return jsonResponse(
                Response.Status.BAD_REQUEST,
                JSONObject().put("error", "No model loaded. Call /load first.")
            )
        }

        // Read threshold and NMS distance from query parameters
        val params = session.parameters
        val threshold = params["threshold"]?.firstOrNull()?.toFloatOrNull() ?: 0.3f
        val distanceThreshold = params["distanceThreshold"]?.firstOrNull()?.toDoubleOrNull() ?: 5.0

        // Read raw image bytes from request body
        val contentLength = session.headers["content-length"]?.toIntOrNull() ?: 0
        if (contentLength == 0) {
            return jsonResponse(
                Response.Status.BAD_REQUEST,
                JSONObject().put("error", "Empty request body. Send image bytes.")
            )
        }
        val imageBytes = ByteArray(contentLength)
        // Use readFully to guarantee all bytes are consumed before decoding
        java.io.DataInputStream(session.inputStream).readFully(imageBytes)

        val bitmap = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
            ?: return jsonResponse(
                Response.Status.BAD_REQUEST,
                JSONObject().put("error", "Failed to decode image from request body.")
            )

        val rawDetections = BuildingDetector.detect(
            bitmap = bitmap,
            clearWeightsAfter = false,
            threshold = threshold
        )
        bitmap.recycle()

        // Apply NMS-like filtering to remove duplicate close detections
        val detections = BuildingDetector.filterCloseDetections(rawDetections, distanceThreshold)

        // Build JSON response
        val detectionsArray = JSONArray()
        for (det in detections) {
            detectionsArray.put(
                JSONObject()
                    .put("x1", det.boundingBox.left)
                    .put("y1", det.boundingBox.top)
                    .put("x2", det.boundingBox.right)
                    .put("y2", det.boundingBox.bottom)
                    .put("score", det.score)
                    .put("classIndex", det.classIndex)
            )
        }

        return jsonResponse(
            Response.Status.OK,
            JSONObject().put("detections", detectionsArray)
        )
    }

    // ── POST /clear ─────────────────────────────────────────────────────────
    private fun handleClear(): Response {
        BuildingDetector.clearWeights()
        return jsonResponse(Response.Status.OK, JSONObject().put("success", true))
    }

    // ── Helpers ─────────────────────────────────────────────────────────────

    private fun readBody(session: IHTTPSession): String {
        val contentLength = session.headers["content-length"]?.toIntOrNull() ?: 0
        val buffer = ByteArray(contentLength)
        session.inputStream.read(buffer, 0, contentLength)
        return String(buffer, Charsets.UTF_8)
    }

    private fun jsonResponse(status: Response.Status, json: JSONObject): Response {
        return newFixedLengthResponse(
            status,
            "application/json",
            json.toString()
        )
    }
}
