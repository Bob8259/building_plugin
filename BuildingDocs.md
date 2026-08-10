# ZKQbuilding — Detector Inference Service for Android

**Version:** 1.11

An Android app that runs a local HTTP server exposing detector inference and text recognition. The app launches a foreground service on startup, making the detection API available to other apps or automation tools on the device via `http://localhost:13462`.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Supported Models](#supported-models)
3. [API Reference](#api-reference)
4. [Text Recognition (PP-OCRv6 small)](#text-recognition-ppocrv6-small)
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

# Run text recognition (self-contained, no load step needed)
curl -X POST http://localhost:13462/ocr \
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
internally by the self-contained [`POST /ocr`](#post-ocr) endpoint:

| Model | Model file | Purpose |
|---|---|---|
| PP-OCRv6 small (det) | `ppocrv6_small_det.onnx` | DB text-line detection |
| PP-OCRv6 small (rec) | `ppocrv6_small_rec.onnx` | CTC text recognition |
| PP-OCRv6 dictionary | `ppocrv6_dict.json` | Character dictionary for CTC decoding |

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
  "ocrLoaded": false
}
```

| Field | Type | Description |
|---|---|---|
| `status` | string | Always `"running"` if the server is up. |
| `version` | string | Current server version. |
| `modelLoaded` | boolean | Whether a model is currently loaded in memory. |
| `modelType` | string \| null | The `modelType` that was used to load the current model, or `null` if none. |
| `ocrLoaded` | boolean | Whether the PP-OCRv6 small det + rec models are currently loaded in memory. |

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

## Text Recognition (PP-OCRv6 small)

Text recognition is exposed as a **self-contained endpoint**: `POST /ocr`.
It requires no load step — the PP-OCRv6 small detection and recognition
models are loaded lazily on the first request and stay resident in memory for
fast subsequent calls. Two optional management endpoints are also provided:
`POST /ocr/load` (pre-warm the models) and `POST /ocr/clear` (free OCR
memory on demand). The models are also released automatically when the
service is destroyed.

### Pipeline

1. **Detection preprocess** — the image long side is scaled to `limit` (dims rounded to multiples of 32) and normalized with ImageNet mean/std in BGR channel order.
2. **DB postprocess** — the probability map is binarized at `detThreshold` (default `0.2`), 8-connected components are extracted, components smaller than 3 px are dropped, boxes whose mean probability is below `boxThreshold` are filtered out, boxes are expanded by `unclipRatio` (default `1.4`), and coordinates are scaled back to the original image space.
3. **Recognition preprocess** — each text box is cropped, resized to height `48` (width clamped to `2..3200`), zero-padded to a minimum width of `recMinWidth` (default `64`), and normalized to `[-1, 1]`.
4. **CTC greedy decode** — index `0` is blank, `dict size + 1` is space, and `1..dict size` map to the characters in `ppocrv6_dict.json`.

### POST /ocr

Run text recognition on a posted image and return the recognized text lines with their bounding boxes. Results are sorted top-to-bottom, then left-to-right.

**Request:**

- **Body:** Raw image bytes (PNG, JPEG, etc. — any format decodable by `BitmapFactory`).
- **Content-Type:** `application/octet-stream` (or any; the server reads raw bytes).
- **Content-Length:** Must be set and non-zero.

**Query parameters:**

All parameters are optional — omitted parameters fall back to their defaults.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `limit` | int | `736` | Long-side limit for the detection input. Larger values improve accuracy on big images at the cost of latency. |
| `boxThreshold` | float | `0.45` | Minimum mean-probability confidence for a text box to be recognized. |
| `detThreshold` | float | `0.2` | DB probability binarization threshold for the text mask. Lower values detect fainter text at the cost of noise. |
| `unclipRatio` | float | `1.4` | Box expansion ratio: `d = w * h * ratio / (2 * (w + h))`. Larger values produce looser boxes around the text. |
| `recMinWidth` | int | `64` | Minimum padded width for the recognition input. Short labels are padded to this width. |

**Example:**

```
POST /ocr?limit=736&boxThreshold=0.45&detThreshold=0.2&unclipRatio=1.4&recMinWidth=64 HTTP/1.1
Content-Length: 54321

<raw image bytes>
```

**Success response:**

```json
{
  "results": [
    {
      "text": "Town Hall 15",
      "score": 0.87,
      "x1": 102,
      "y1": 200,
      "x2": 350,
      "y2": 232
    }
  ]
}
```

| Field | Type | Description |
|---|---|---|
| `text` | string | Recognized text content of the line. |
| `score` | float | Detection confidence — mean DB probability of the text box (0–1). |
| `x1`, `y1` | float | Top-left corner of the text box (original image pixel coordinates). |
| `x2`, `y2` | float | Bottom-right corner of the text box. |

**Error responses:**

| Status | Condition |
|---|---|
| 400 | Empty request body. |
| 400 | Image bytes could not be decoded. |
| 500 | Model loading or inference failed (message included). |

---

### POST /ocr/load

Optional warm-up: pre-load the PP-OCRv6 small det + rec models so the first
`/ocr` call does not pay the model-loading latency. No-op if already loaded.

**Request body:** *(empty or ignored)*

**Success response:**

```json
{ "success": true }
```

**Error response (500) — models failed to load:**

```json
{ "success": false, "error": "..." }
```

---

### POST /ocr/clear

Unload the PP-OCRv6 small det + rec models and free their memory. The next
`/ocr` call will lazily reload them.

**Request body:** *(empty or ignored)*

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

# Load a model
requests.post(f"{BASE}/load", json={"modelType": "capital-building-detect"})

# Detect objects in a screenshot
with open("screenshot.png", "rb") as f:
    r = requests.post(
        f"{BASE}/detect",
        params={"threshold": 0.4, "distanceThreshold": 8},
        data=f.read(),
        headers={"Content-Type": "application/octet-stream"},
    )
print(r.json()["detections"])

# Unload model when done
requests.post(f"{BASE}/clear")

# Text recognition — self-contained, no load step needed
# All query params are optional; omitted ones use defaults:
#   limit=736, boxThreshold=0.45, detThreshold=0.2, unclipRatio=1.4, recMinWidth=64
with open("screenshot.png", "rb") as f:
    r = requests.post(
        f"{BASE}/ocr",
        params={"limit": 736, "boxThreshold": 0.45},
        data=f.read(),
        headers={"Content-Type": "application/octet-stream"},
    )
for line in r.json()["results"]:
    print(line["text"], line["score"], line["x1"], line["y1"], line["x2"], line["y2"])

# Optional: pre-warm OCR models before the first /ocr call
requests.post(f"{BASE}/ocr/load")

# Optional: free OCR memory when done
requests.post(f"{BASE}/ocr/clear")
```

### ADB + curl (on-device)

```bash
# Forward port from PC to device
adb forward tcp:13462 tcp:13462

# Now use curl from your PC
curl http://localhost:13462/status
curl -X POST http://localhost:13462/load -d '{"modelType":"numbers"}'
curl -X POST http://localhost:13462/detect \
  -H "Content-Type: application/octet-stream" \
  --data-binary @image.png

# Text recognition
curl -X POST http://localhost:13462/ocr \
  -H "Content-Type: application/octet-stream" \
  --data-binary @image.png

# Optional OCR model management
curl -X POST http://localhost:13462/ocr/load
curl -X POST http://localhost:13462/ocr/clear
```
