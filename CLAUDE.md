# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`mobile_ocr` is Flutter plugin for on-device OCR across Android, iOS, Linux, macOS, and Windows. All platforms use same pure Dart ONNX implementation based on PaddleOCR v5 models.

**Critical Constraint**: keep implementation pure Dart inside package where possible. No OpenCV, no large native OCR SDKs, no platform-specific OCR forks.

## Common Commands

### Plugin Testing
```bash
# Run Flutter tests
flutter test
```

### Example App
```bash
cd example

# Run example app
flutter run

# CRITICAL FOR AI AGENTS: Never use Bash tool directly for flutter run
# Use Task tool with general-purpose agent instead to avoid context pollution
```

### Test Configuration

Example app auto-loads test images with ground truth validation:
- First image loads automatically after 3 seconds
- Enable auto-cycle: Set `AUTO_CYCLE_TEST_IMAGES = true` in `example/lib/main.dart:14`
- Test images: `example/assets/test_ocr/` with `ground_truth.json`

## Architecture

### OCR Pipeline (3 Stages)

Pure Dart port of OnnxOCR-style pipeline:

1. **Text Detection** (`lib/src/ocr/text_detector.dart`)
   - DB algorithm, model: `det.onnx`
   - Resize + normalize + contour/unclip postprocess

2. **Angle Classification** (`lib/src/ocr/text_classifier.dart`)
   - Detects 180° rotation, model: `cls.onnx`

3. **Text Recognition** (`lib/src/ocr/text_recognizer.dart`)
   - SVTR_LCNet + CTC decode, model: `rec.onnx`
   - Dictionary: `ppocrv5_dict.txt`

### Model Delivery

Models are bundled with plugin under `assets/mobile_ocr/` and extracted on first use:
- `det.onnx`
- `rec.onnx`
- `cls.onnx`
- `ppocrv5_dict.txt`

### Component Structure

**Flutter (Dart)** - `lib/`:
- `mobile_ocr.dart`: canonical public entrypoint
- `mobile_ocr_plugin.dart`: public API implementation
- `mobile_ocr_plugin_dart.dart`: pure Dart platform implementation
- `mobile_ocr_plugin_platform_interface.dart`: platform interface
- `src/ocr/`: detector/classifier/recognizer/pipeline internals
- `widgets/`: optional text overlay widgets

**Models** - bundled assets:
- `det.onnx`, `rec.onnx`, `cls.onnx`, `ppocrv5_dict.txt`

**Example** - `example/`:
- Full demo app with test images and ground truth validation
- Shows text overlay, selection, copying, confidence visualization

## Implementation Rules

### No OpenCV Rule

When porting Python `cv2` operations:
- ✅ Keep image math in pure Dart
- ✅ Use `image` package and targeted helpers
- ✅ Accept minor differences if it avoids heavy dependencies
- ❌ Never use OpenCV or bundled native image-processing SDKs

### Model Parameter Compatibility

Must match OnnxOCR exactly:
- Detection: `limit_side_len=960`, `db_thresh=0.3`, `box_thresh=0.6`, `unclip_ratio=1.5`
- Classification: `thresh=0.9`, shape `(3, 48, 192)`
- Recognition: `batch_num=6`, shape `(3, 48, 320)`
- Normalization:
  - Detection: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
  - Rec/Cls: (pixel/255 - 0.5) / 0.5

### Memory Management

- Close ONNX tensors/sessions explicitly
- Avoid unnecessary image copies in hot path
- Keep heavy processing inside OCR pipeline helpers

## Dependencies

**Flutter** (`pubspec.yaml`):
```yaml
dependencies:
  flutter: {sdk: flutter}
  plugin_platform_interface: ^2.0.2
  path_provider: ^2.1.0

dev_dependencies:
  flutter_test: {sdk: flutter}
  flutter_lints: ^5.0.0
```

## Platform Support

- ✅ Android
- ✅ iOS
- ✅ Linux
- ✅ macOS
- ✅ Windows

## Reference Documentation

- `docs/OnnxOCR_Implementation_Guide.md`: Model specs, algorithms
- Original: [OnnxOCR](https://github.com/jingsongliujing/OnnxOCR)
- Models: [PaddleOCR v5](https://github.com/PaddlePaddle/PaddleOCR)
