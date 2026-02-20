import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:path_provider/path_provider.dart';
import '../mobile_ocr_plugin_platform_interface.dart';
import '../src/ocr/ocr_processor.dart';
import '../src/ocr/types.dart';
import '../src/ocr/fast_image_loader.dart';

class DartMobileOcr extends MobileOcrPlatform {
  OcrProcessor? _processor;
  String? _modelPath;
  bool _isInitialized = false;
  bool _isInitializing = false;

  static const String _modelVersion = 'pp-ocrv5-202410';
  static const List<String> _modelFiles = [
    'det.onnx',
    'rec.onnx',
    'cls.onnx',
    'ppocrv5_dict.txt',
  ];

  static Future<String> _getModelsDirectory() async {
    final appDir = await getApplicationSupportDirectory();
    return '${appDir.path}/assets/mobile_ocr';
  }

  Future<void> _extractModelsFromAssets(String modelsDir) async {
    final dir = Directory(modelsDir);
    if (!await dir.exists()) {
      await dir.create(recursive: true);
    }

    for (final modelFile in _modelFiles) {
      final targetFile = File('$modelsDir/$modelFile');
      if (!await targetFile.exists()) {
        try {
          final data = await rootBundle.load(
            'packages/mobile_ocr/assets/mobile_ocr/$modelFile',
          );
          await targetFile.writeAsBytes(
            data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes),
            flush: true,
          );
        } catch (_) {
          // Model file not in assets, skip (may be optional like cls.onnx)
        }
      }
    }
  }

  Future<void> _ensureInitialized() async {
    if (_isInitialized) return;
    if (_isInitializing) {
      while (_isInitializing) {
        await Future.delayed(const Duration(milliseconds: 10));
      }
      return;
    }

    _isInitializing = true;
    try {
      final modelsDir = await _getModelsDirectory();
      _modelPath = modelsDir;

      await _extractModelsFromAssets(modelsDir);

      final detectionModel = '$modelsDir/det.onnx';
      final recognitionModel = '$modelsDir/rec.onnx';
      final classificationModel = '$modelsDir/cls.onnx';
      final dictionaryPath = '$modelsDir/ppocrv5_dict.txt';

      final detExists = await File(detectionModel).exists();
      final recExists = await File(recognitionModel).exists();
      final dictExists = await File(dictionaryPath).exists();

      if (!detExists || !recExists || !dictExists) {
        throw StateError(
          'Models not found at $modelsDir. '
          'Please ensure models are bundled with the application.',
        );
      }

      _processor = await OcrProcessor.create(
        detectionModelPath: detectionModel,
        recognitionModelPath: recognitionModel,
        classificationModelPath: await File(classificationModel).exists()
            ? classificationModel
            : null,
        dictionaryPath: dictionaryPath,
        useAngleClassification: await File(classificationModel).exists(),
      );

      _isInitialized = true;
    } finally {
      _isInitializing = false;
    }
  }

  @override
  Future<String?> getPlatformVersion() async {
    return '${Platform.operatingSystem} ${Platform.operatingSystemVersion}';
  }

  @override
  Future<Map<dynamic, dynamic>> prepareModels() async {
    try {
      await _ensureInitialized();
      return {
        'isReady': true,
        'version': _modelVersion,
        'modelPath': _modelPath,
      };
    } catch (e) {
      return {
        'isReady': false,
        'version': null,
        'modelPath': null,
        'error': e.toString(),
      };
    }
  }

  @override
  Future<List<Map<dynamic, dynamic>>> detectText({
    required String imagePath,
    bool includeAllConfidenceScores = false,
  }) async {
    await _ensureInitialized();

    final file = File(imagePath);
    if (!await file.exists()) {
      throw ArgumentError('Image file does not exist: $imagePath');
    }

    final image = await _loadAndConvertImage(imagePath);
    if (image == null) {
      throw ArgumentError('Could not decode image: $imagePath');
    }

    final result = await _processor!.processImage(
      image,
      includeAllConfidenceScores: includeAllConfidenceScores,
    );

    return _convertResultToMap(result);
  }

  @override
  Future<bool> hasText({required String imagePath}) async {
    await _ensureInitialized();

    final file = File(imagePath);
    if (!await file.exists()) {
      throw ArgumentError('Image file does not exist: $imagePath');
    }

    final image = await _loadAndConvertImage(imagePath);
    if (image == null) {
      throw ArgumentError('Could not decode image: $imagePath');
    }

    final result = await _processor!.hasHighConfidenceText(image);
    return result.hasText;
  }

  Future<img.Image?> _loadAndConvertImage(String imagePath) async {
    final file = File(imagePath);
    final bytes = await file.readAsBytes();

    final image = await FastImageLoader.loadFromBytes(bytes);
    if (image == null) {
      return img.decodeImage(bytes);
    }
    return image;
  }

  List<Map<dynamic, dynamic>> _convertResultToMap(OcrResult result) {
    final List<Map<dynamic, dynamic>> maps = [];

    for (int i = 0; i < result.boxes.length; i++) {
      final box = result.boxes[i];
      final text = result.texts[i];
      final confidence = result.scores[i];
      final characters = result.characters[i];

      maps.add({
        'text': text,
        'confidence': confidence,
        'points': box.points.map((p) => {'x': p.x, 'y': p.y}).toList(),
        'boundingBox': {
          'left': box.boundingRect().left,
          'top': box.boundingRect().top,
          'right': box.boundingRect().right,
          'bottom': box.boundingRect().bottom,
        },
        'characters': characters
            .map(
              (c) => {
                'text': c.text,
                'confidence': c.confidence,
                'points': c.points.map((p) => {'x': p.x, 'y': p.y}).toList(),
              },
            )
            .toList(),
      });
    }

    return maps;
  }

  void dispose() {
    _processor?.close();
    _processor = null;
    _isInitialized = false;
  }
}
