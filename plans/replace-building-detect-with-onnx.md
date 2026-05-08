# Plan: Replace building-detect TFLite with ONNX

## Summary

Route `"building-detect"` through `OnnxRuntime` (loading `my_building_detector.onnx`) instead of `TfliteRuntime`. Delete the unused `.tflite` file and remove the `"building-detect-onnx"` model type that was added previously.

## Steps

1. **Delete** `app/src/main/assets/my_building_detector.tflite`

2. **`BuildingDetector.kt` line 39** — change condition:
   - From: `if (modelType == "building-detect-onnx")`
   - To: `if (modelType == "building-detect")`

3. **`TfliteRuntime.kt` line 95** — remove the `"building-detect"` branch from the `when` block (it no longer maps to a TFLite file)

4. **`TfliteRuntime.kt` line 101** — remove `building-detect-onnx` from the error message valid types list

5. **`DetectorHttpServer.kt` line 69** — remove `building-detect-onnx` from the valid types error message
