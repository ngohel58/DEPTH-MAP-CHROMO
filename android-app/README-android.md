# DepthMap ONNX Android App

This module contains a simple Android application that mirrors the functionality of `index.html`. The app allows selecting an image, generating a depth map using ONNXRuntime Mobile and the MiDaS model, and viewing a color mapped result.

## Requirements
- Android Studio Hedgehog or later
- Android SDK 33
- Kotlin 1.8+
- Gradle 7+

## Setting up the Model
The MiDaS ONNX model is not included in the repository. Download the lightweight model:

```
https://github.com/isl-org/MiDaS/releases/download/v3_1_small/model-small.onnx
```

Rename the file to `midas_small.onnx` and place it in `app/src/main/assets/`.

You can substitute other depth models compatible with ONNXRuntime (e.g., Depth Anything V2 or Marigold). Ensure the input preprocessing matches the model's requirements.

## Building
Open the `android-app` folder in Android Studio and build/run the `app` module on a device running Android 7.0+.

## Usage
1. Tap **Select Photo** to choose an image from the gallery.
2. Tap **Generate Depth Map** to run the model and display the colored depth result.

This example shows a minimal pipeline. You can extend it with additional color map options or more complex preprocessing for different models.
