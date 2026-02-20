import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';
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

  static Future<String> _getModelsDirectory() async {
    final appDir = await getApplicationSupportDirectory();
    return '${appDir.path}/assets/mobile_ocr';
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
    return 'Linux ${Platform.operatingSystemVersion}';
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
    Uint8List bytes = await file.readAsBytes();

    final ext = imagePath.toLowerCase().split('.').last;
    final isHeic = ext == 'heic' || ext == 'heif';

    if (isHeic) {
      final tempDir = await getTemporaryDirectory();
      final outputPath =
          '${tempDir.path}/converted_${DateTime.now().millisecondsSinceEpoch}.png';

      try {
        final result = await Process.run(
          'heif-convert',
          [imagePath, outputPath],
          stdoutEncoding: utf8,
          stderrEncoding: utf8,
        );

        if (result.exitCode != 0) {
          throw Exception(
            'HEIC conversion failed: ${result.stderr}\n'
            'Please install heif-examples package (sudo apt install heif-examples)',
          );
        }

        final convertedFile = File(outputPath);
        if (await convertedFile.exists()) {
          bytes = await convertedFile.readAsBytes();
          await convertedFile.delete();
        } else {
          throw Exception('HEIC conversion failed: output file not created');
        }
      } catch (e) {
        if (e is Exception) rethrow;
        throw Exception(
          'HEIC conversion error: $e\n'
          'Please install heif-examples package (sudo apt install heif-examples)',
        );
      }
    }

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
