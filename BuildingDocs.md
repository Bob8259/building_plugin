# ZKQbuilding — Detector Inference Service for Android

**Version:** 1.11

An Android app that runs a local HTTP server exposing detector inference and text recognition (PP-OCRv6 Small, FP16 & FP32). The app launches a foreground service on startup, making the detection API available to other apps or automation tools on the device via `http://localhost:13462`.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Supported Models](#supported-models)
3. [API Reference](#api-reference)
4. [Text Recognition (PP-OCRv6 Small)](#text-recognition-ppocrv6-small)
5. [Usage Examples](#usage-examples)

---

## Quick Start

1. Install the APK on the target device.
2. Launch the app once — the foreground service starts automatically.
3. The HTTP server is now listening on **port 13462**.
4. Load a model, then send images for detection.

```
# Verify the service is running
curl http://localhost:13462/status

# Load a model (modelType is required)
curl -X POST http://localhost:13462/load -d '{"modelType":"capital-building-detect"}'

# Run detection on an image
curl -X POST http://localhost:13462/detect \
  --data-binary @screenshot.png

# Run text recognition (defaults to FP16, self-contained, no load step needed)
curl -X POST http://localhost:13462/ocr \
  --data-binary @screenshot.png

# Run text recognition with FP32 precision
curl -X POST "http://localhost:13462/ocr?precision=fp32" \
  --data-binary @screenshot.png
```

You can also start the service without opening the app UI:

```
adb shell am start-foreground-service -n com.building.plugin/.service.DetectorService
```

---

## Supported Models

Models are loaded by passing a `modelType` string to the `/load` endpoint.
All detector models are bundled in the APK assets directory. On app start, every
`.onnx` asset is extracted into the app's private storage (`filesDir/models`),
overwriting any existing copies, and models are loaded from that private path.
If a model file is missing from private storage at load time, the app
automatically re-extracts all models once; if it is still missing, `/load`
returns an error message.

| `modelType` value | Model file | Source |
|---|---|---|
| `"walls-detect"` | `walls_detector.onnx` | Assets → extracted to private storage on startup |
| `"numbers"` | `numbers_detector.onnx` | Assets → extracted to private storage on startup |
| `"building-detect"` | `my_building_detector.onnx` | Assets → extracted to private storage on startup |
| `"capital-building-detect"` | `capital_building_detector.onnx` | Assets → extracted to private storage on startup |
| `"remove-obstacle"` | `obstacles_detector.onnx` | Assets → extracted to private storage on startup |
| `"clan-war-numbers"` | `clan_war_number_detector.onnx` | Assets → extracted to private storage on startup |
| `"clan-game"` | `clan_game_detector.onnx` | Assets → extracted to private storage on startup |
| `"main-base-battle"` | `main_base_battle.onnx` | Assets → extracted to private storage on startup |

The text recognition models are **not** loaded via `/load` — they are managed
internally by the self-contained [`POST /ocr`](#post-ocr) endpoint. Both FP16
(recommended for mobile efficiency) and FP32 precisions are bundled:

| Model | Model file | Precision | Purpose |
|---|---|---|---|
| PP-OCRv6 Small (det fp16) | `det_fp16.onnx` | FP16 | DB text-line detection (~5.0 MB) |
| PP-OCRv6 Small (rec fp16) | `rec_fp16.onnx` | FP16 | CTC text recognition (~10.6 MB) |
| PP-OCRv6 Small (det fp32) | `det_fp32.onnx` | FP32 | DB text-line detection (~9.9 MB) |
| PP-OCRv6 Small (rec fp32) | `rec_fp32.onnx` | FP32 | CTC text recognition (~21.2 MB) |
| PP-OCRv6 character list | `rec_characters.json` | — | Character dictionary for CTC decoding |

> **Note:** `modelType` is required. Omitting it or passing an unrecognized value will result in an error.
> If a model asset has not been packaged under `app/src/main/assets`, `/load` will fail with a missing-model error.

The detector dynamically reads the model's input/output tensor shapes, so any compatible object-detection TFLite model with output shape `[1, N, 6]` (where each detection is `[x1, y1, x2, y2, score, classIndex]`) will work.

---

## API Reference

Base URL: `http://localhost:13462`

All responses are JSON with `Content-Type: application/json`.

### GET /status

Health check. Returns the server status, version, and current model state.

**Response:**

```json
{
  "status": "running",
  "version": "1.11",
  "modelLoaded": true,
  "modelType": "walls-detect",
  "ocrLoaded": false,
  "ocrPrecision": null
}
```

| Field | Type | Description |
|---|---|---|
| `status` | string | Always `"running"` if the server is up. |
| `version` | string | Current server version. |
| `modelLoaded` | boolean | Whether a model is currently loaded in memory. |
| `modelType` | string \| null | The `modelType` that was used to load the current model, or `null` if none. |
| `ocrLoaded` | boolean | Whether PP-OCRv6 Small det + rec models are currently loaded in memory. |
| `ocrPrecision` | string \| null | Active OCR model precision (`"fp16"`, `"fp32"`), or `null` if not loaded. |

---

### POST /load

Load model weights into the TFLite interpreter. If the same model type is already loaded, this is a no-op.

**Request body (JSON):**

```json
{
  "modelType": "walls-detect"
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `modelType` | string | **Yes** | Model identifier (see [Supported Models](#supported-models)). Must be one of the listed values. |

**Success response:**

```json
{ "success": true }
```

**Error response (400) — missing or invalid `modelType`:**

```json
{ "success": false, "error": "\"modelType\" is required. Valid types: walls-detect, numbers, building-detect, capital-building-detect, remove-obstacle, clan-war-numbers, clan-game, main-base-battle" }
```

**Error response (500) — model failed to load:**

```json
{ "success": false, "error": "..." }
```

---

### POST /detect

Run object detection on a posted image. A model **must** be loaded first via `/load`.

**Request:**

- **Body:** Raw image bytes (PNG, JPEG, etc. — any format decodable by `BitmapFactory`).
- **Content-Type:** `application/octet-stream` (or any; the server reads raw bytes).
- **Content-Length:** Must be set and non-zero.

**Query parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `threshold` | float | `0.3` | Minimum confidence score to include a detection. |
| `distanceThreshold` | double | `5.0` | Pixel distance below which two detections are considered duplicates (NMS-like filtering). The higher-confidence detection is kept. |

**Example:**

```
POST /detect?threshold=0.5&distanceThreshold=10 HTTP/1.1
Content-Length: 54321

<raw image bytes>
```

**Success response:**

```json
{
  "detections": [
    {
      "x1": 102.5,
      "y1": 200.0,
      "x2": 350.0,
      "y2": 410.3,
      "score": 0.92,
      "classIndex": 0
    }
  ]
}
```

| Field | Type | Description |
|---|---|---|
| `x1`, `y1` | float | Top-left corner of the bounding box (in original image pixel coordinates). |
| `x2`, `y2` | float | Bottom-right corner of the bounding box. |
| `score` | float | Detection confidence (0–1). |
| `classIndex` | int | Predicted class index. |

**Error responses:**

| Status | Condition |
|---|---|
| 400 | No model loaded — call `/load` first. |
| 400 | Empty request body. |
| 400 | Image bytes could not be decoded. |

---

### POST /clear

Unload model weights and free the TFLite interpreter.

**Request body:** *(empty or ignored)*

**Success response:**

```json
{ "success": true }
```

---

## Text Recognition (PP-OCRv6 Small)

Text recognition is exposed as a **self-contained endpoint**: `POST /ocr`.
It requires no explicit load step — the detection and recognition models are
loaded lazily on the first request and stay resident in memory for fast
subsequent calls. Two precisions are supported: **FP16** (default, ~15.6 MB
combined, optimized for mobile efficiency) and **FP32** (~31.1 MB combined).

Two optional management endpoints are also provided: `POST /ocr/load`
(pre-warm models) and `POST /ocr/clear` (free OCR memory on demand). Models
are also released automatically when the foreground service is destroyed.

### Pipeline

1. **Detection preprocess** — the image long side is scaled to `limit` (dims rounded to multiples of 32, max 4000) and normalized with ImageNet mean/std in BGR channel order.
2. **DB postprocess** — the probability map is binarized at `detThreshold` (default `0.20`), 8-connected components are extracted, boxes below `boxThreshold` (default `0.45`) are filtered out, expanded by `unclipRatio` (default `1.4`), and scaled back to original image space.
3. **Reading order sorting** — text boxes are grouped into horizontal rows and sorted top-to-bottom, left-to-right.
4. **Recognition preprocess** — each text box is cropped, resized to height `48` (width clamped to `2..3200`), zero-padded to a minimum width of `recMinWidth` (default `320`), and normalized to `[-1, 1]` in RGB channel order.
5. **CTC greedy decode** — index `0` is blank, index `1..N` map to characters in `rec_characters.json`, and index `N + 1` is space (`" "`).

### POST /ocr

Run text recognition on a posted image and return the recognized text lines with their bounding boxes, detection scores, and recognition confidences.

**Request:**

- **Body:** Raw image bytes (PNG, JPEG, etc. — any format decodable by `BitmapFactory`).
- **Content-Type:** `application/octet-stream` (or any; the server reads raw bytes).
- **Content-Length:** Must be set and non-zero.

**Query parameters (all optional):**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `precision` | string | `"fp16"` | Model precision to use: `"fp16"` (recommended for mobile) or `"fp32"`. |
| `limit` | int | `960` | Long-side limit for detection input. |
| `boxThreshold` | float | `0.45` | Minimum mean-probability confidence for a text box. |
| `detThreshold` | float | `0.20` | DB probability binarization threshold. |
| `unclipRatio` | float | `1.4` | Box expansion ratio: `d = w * h * ratio / (2 * (w + h))`. |
| `recMinWidth` | int | `320` | Minimum padded width for recognition input. |

**Example:**

```
POST /ocr?precision=fp16&limit=960&boxThreshold=0.45 HTTP/1.1
Content-Length: 54321

<raw image bytes>
```

**Success response:**

```json
{
  "precision": "fp16",
  "results": [
    {
      "text": "Town Hall 15",
      "score": 0.88,
      "confidence": 0.96,
      "x1": 102.0,
      "y1": 200.0,
      "x2": 350.0,
      "y2": 232.0
    }
  ]
}
```

| Field | Type | Description |
|---|---|---|
| `precision` | string | The precision used for inference (`"fp16"` or `"fp32"`). |
| `text` | string | Recognized text content of the line. |
| `score` | float | Detection confidence — mean DB probability of the box (0–1). |
| `confidence` | float | Recognition confidence — mean CTC confidence of the characters (0–1). |
| `x1`, `y1` | float | Top-left corner of the text box (original image coordinates). |
| `x2`, `y2` | float | Bottom-right corner of the text box. |

---

### POST /ocr/load

Optional warm-up: pre-loads det + rec models into memory. No-op if already loaded with the same precision.

**Query parameter or JSON body:**
- `precision`: `"fp16"` (default) or `"fp32"`

**Success response:**

```json
{
  "success": true,
  "precision": "fp16"
}
```

---

### POST /ocr/clear

Unload the active PP-OCRv6 Small models and free their memory. The next `/ocr` call will lazily reload them.

**Success response:**

```json
{ "success": true }
```

---

## Usage Examples

### Python

```python
import requests

BASE = "http://localhost:13462"

# Check status
r = requests.get(f"{BASE}/status")
print(r.json())

# 1. Object Detection
requests.post(f"{BASE}/load", json={"modelType": "capital-building-detect"})
with open("screenshot.png", "rb") as f:
    r = requests.post(
        f"{BASE}/detect",
        params={"threshold": 0.4, "distanceThreshold": 8},
        data=f.read(),
        headers={"Content-Type": "application/octet-stream"},
    )
print(r.json()["detections"])
requests.post(f"{BASE}/clear")

# 2. Text Recognition (FP16 by default)
with open("screenshot.png", "rb") as f:
    r = requests.post(
        f"{BASE}/ocr",
        params={"precision": "fp16", "limit": 960},
        data=f.read(),
        headers={"Content-Type": "application/octet-stream"},
    )
for line in r.json()["results"]:
    print(line["text"], line["score"], line["confidence"], line["x1"], line["y1"])

# Optional: pre-warm OCR models
requests.post(f"{BASE}/ocr/load", params={"precision": "fp16"})

# Optional: free OCR memory when done
requests.post(f"{BASE}/ocr/clear")
```

### ADB + curl (on-device)

```bash
# Forward port from PC to device
adb forward tcp:13462 tcp:13462

# Check status
curl http://localhost:13462/status

# Object detection
curl -X POST http://localhost:13462/load -d '{"modelType":"numbers"}'
curl -X POST http://localhost:13462/detect \
  -H "Content-Type: application/octet-stream" \
  --data-binary @image.png
curl -X POST http://localhost:13462/clear

# Text recognition (FP16 default)
curl -X POST http://localhost:13462/ocr \
  -H "Content-Type: application/octet-stream" \
  --data-binary @image.png

# Text recognition (FP32)
curl -X POST "http://localhost:13462/ocr?precision=fp32" \
  -H "Content-Type: application/octet-stream" \
  --data-binary @image.png

# Optional OCR model management
curl -X POST "http://localhost:13462/ocr/load?precision=fp16"
curl -X POST http://localhost:13462/ocr/clear
```
