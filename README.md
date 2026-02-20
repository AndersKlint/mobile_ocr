# Mobile OCR

Mobile OCR is a Flutter plugin that delivers fully on-device text detection and
recognition across Android, iOS, Linux, macOS, and Windows. All platforms share
the same Dart ONNX implementation using PaddleOCR v5 models.

## Features

- Text detection (DB algorithm) with oriented bounding polygons
- Text recognition (SVTR_LCNet + CTC) mirroring PaddleOCR v5 behaviour
- Text angle classification and auto-rotation for skewed crops
- On-device processing with no network calls
- Multi-language character dictionary (Chinese + English)
- Pure Dart implementation - same code across all platforms

## Installation

Add this to your package's `pubspec.yaml` file:

```yaml
dependencies:
  mobile_ocr:
    git:
      url: https://github.com/AndersKlint/mobile_ocr
```

## Usage

### Basic Usage

```dart
import 'package:mobile_ocr/mobile_ocr_plugin.dart';

// Create plugin instance
final ocrPlugin = MobileOcr();

// Ensure ONNX models are extracted (done automatically on first use)
await ocrPlugin.prepareModels();

// Optional quick check if the image contains high-confidence text
// (runs much faster than full text recognition)
final hasText = await ocrPlugin.hasText(
  imagePath: '/path/to/image.png',
);

// Perform OCR by supplying an image path
final textBlocks = await ocrPlugin.detectText(
  imagePath: '/path/to/image.png',
);

for (final block in textBlocks) {
  print('Text: ${block.text}');
  print('Confidence: ${block.confidence}');
  print('Corners: ${block.points}');
  final bounds = block.boundingBox;
  print('Bounds: ${bounds.left}, ${bounds.top} -> ${bounds.right}, ${bounds.bottom}');
}
```

#### Detection Output

Each `TextBlock` mirrors the shape produced by the PaddleOCR detector:

- `text` – recognized string
- `confidence` – recognition probability (0–1)
- `points` – four corner points (clockwise) describing the oriented quadrilateral
- `boundingBox` – convenience `Rect` derived from the polygon for quick overlays

### Using with Image Picker

```dart
import 'package:image_picker/image_picker.dart';

final ImagePicker picker = ImagePicker();
final XFile? image = await picker.pickImage(source: ImageSource.gallery);

if (image != null) {
  final result = await ocrPlugin.detectText(imagePath: image.path);
  // Process results...
}
```

## Example App

The plugin includes a comprehensive example app that demonstrates:

- Loading images from camera or gallery
- Running OCR on selected images
- Displaying detected text regions with colored overlays
- Tapping on text regions to view and copy the recognized text

To run the example:

```bash
cd example
flutter run
```

## Model Assets

The ONNX models (~21 MB total) are bundled with the plugin in `assets/mobile_ocr/`:

- `det.onnx` (4.6 MB) - Text detection model
- `rec.onnx` (16 MB) - Text recognition model
- `cls.onnx` (570 KB) - Text angle classification model
- `ppocrv5_dict.txt` (73 KB) - Character dictionary

On first use, `prepareModels()` extracts them to the application support directory.
This ensures the app works offline immediately after installation.

## Platform Support

Currently supports:

- ✅ Android (API 24+)
- ✅ iOS 14+
- ✅ Linux
- ✅ macOS
- ✅ Windows

All platforms use the same pure Dart ONNX implementation via `onnxruntime_v2`.

## Acknowledgments

This work would not be possible without:

- [Ente Mobile OCR](https://github.com/ente-io/mobile_ocr) - Original plugin this was forked from
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) - The original OCR models and algorithms
- [OnnxOCR](https://github.com/jingsongliujing/OnnxOCR) - ONNX implementation and pipeline architecture

## License

This plugin is released under the MIT License. The ONNX models are derived from PaddleOCR and follow their licensing terms.
