import 'dart:async';
import 'dart:io';
import 'package:flutter/foundation.dart';
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
  _ProcessorConfiguration? _processorConfiguration;
  Future<void>? _initializationFuture;
  Future<void> _operationQueue = Future<void>.value();

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

    final assetPaths = [
      'packages/mobile_ocr/assets/mobile_ocr',
      'assets/mobile_ocr',
    ];

    for (final modelFile in _modelFiles) {
      final targetFile = File('$modelsDir/$modelFile');
      if (await targetFile.exists()) continue;

      Object? lastError;
      bool extracted = false;

      for (final assetPath in assetPaths) {
        try {
          final fullPath = '$assetPath/$modelFile';
          debugPrint('Mobile OCR: Trying to load asset: $fullPath');
          final data = await rootBundle.load(fullPath);
          await targetFile.writeAsBytes(
            data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes),
            flush: true,
          );
          debugPrint(
            'Mobile OCR: Extracted $modelFile (${data.lengthInBytes} bytes)',
          );
          extracted = true;
          break;
        } catch (e) {
          lastError = e;
          debugPrint(
            'Mobile OCR: Failed to load from $assetPath/$modelFile: $e',
          );
        }
      }

      if (!extracted && modelFile != 'cls.onnx') {
        throw StateError(
          'Failed to extract $modelFile. Last error: $lastError. '
          'Tried paths: ${assetPaths.map((p) => '$p/$modelFile').join(", ")}. '
          'Ensure assets are declared in your pubspec.yaml.',
        );
      }
    }
  }

  Future<void> _ensureInitialized() async {
    await _ensureInitializedWithOptions(
      debugDumpDir: null,
      processingOptions: const OcrProcessingOptions(),
    );
  }

  Future<void> _ensureInitializedWithOptions({
    required String? debugDumpDir,
    required OcrProcessingOptions processingOptions,
  }) async {
    final requestedConfiguration = _ProcessorConfiguration(
      debugDumpDir: debugDumpDir,
      processingOptions: processingOptions,
    );
    if (_isProcessorReadyFor(requestedConfiguration)) {
      return;
    }

    final initializationFuture = _initializationFuture ??= _initializeProcessor(
      requestedConfiguration,
    );
    try {
      await initializationFuture;
    } finally {
      if (identical(_initializationFuture, initializationFuture)) {
        _initializationFuture = null;
      }
    }
  }

  bool _isProcessorReadyFor(_ProcessorConfiguration configuration) {
    return _processor != null && _processorConfiguration == configuration;
  }

  Future<void> _initializeProcessor(
    _ProcessorConfiguration configuration,
  ) async {
    if (_processor != null && _processorConfiguration != configuration) {
      await _disposeProcessor();
    }

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
    final hasClassificationModel = await File(classificationModel).exists();

    if (!detExists || !recExists || !dictExists) {
      throw StateError(
        'Models not found at $modelsDir. '
        'Please ensure models are bundled with the application.',
      );
    }

    _processor = await OcrProcessor.create(
      detectionModelPath: detectionModel,
      recognitionModelPath: recognitionModel,
      classificationModelPath: hasClassificationModel
          ? classificationModel
          : null,
      dictionaryPath: dictionaryPath,
      useAngleClassification: hasClassificationModel,
      debugDumpDir: configuration.debugDumpDir,
      processingOptions: configuration.processingOptions,
    );
    _processorConfiguration = configuration;
  }

  Future<void> _disposeProcessor() async {
    final processor = _processor;
    _processor = null;
    _processorConfiguration = null;
    if (processor != null) {
      await processor.close();
    }
  }

  OcrProcessingOptions _buildProcessingOptions({
    required bool trimRecognitionWhitespace,
    required bool enhanceRecognitionCrops,
    required double recognitionContrastBoost,
    required double recognitionBrightnessBoost,
  }) {
    return OcrProcessingOptions(
      trimRecognitionWhitespace: trimRecognitionWhitespace,
      enhanceRecognitionCrops: enhanceRecognitionCrops,
      recognitionContrastBoost: recognitionContrastBoost,
      recognitionBrightnessBoost: recognitionBrightnessBoost,
    );
  }

  Future<T> _runExclusive<T>(Future<T> Function() action) {
    final previousOperation = _operationQueue;
    final nextOperation = Completer<void>();
    _operationQueue = nextOperation.future;

    return previousOperation.then((_) => action()).whenComplete(() {
      if (!nextOperation.isCompleted) {
        nextOperation.complete();
      }
    });
  }

  @override
  Future<String?> getPlatformVersion() async {
    return '${Platform.operatingSystem} ${Platform.operatingSystemVersion}';
  }

  @override
  Future<Map<dynamic, dynamic>> prepareModels() async {
    return _runExclusive(() async {
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
    });
  }

  @override
  Future<List<Map<dynamic, dynamic>>> detectText({
    required String imagePath,
    bool includeAllConfidenceScores = false,
    String? debugDumpDir,
    bool trimRecognitionWhitespace = false,
    bool enhanceRecognitionCrops = false,
    double recognitionContrastBoost = 0.08,
    double recognitionBrightnessBoost = 0.02,
  }) async {
    return _runExclusive(() async {
      await _ensureInitializedWithOptions(
        debugDumpDir: debugDumpDir,
        processingOptions: _buildProcessingOptions(
          trimRecognitionWhitespace: trimRecognitionWhitespace,
          enhanceRecognitionCrops: enhanceRecognitionCrops,
          recognitionContrastBoost: recognitionContrastBoost,
          recognitionBrightnessBoost: recognitionBrightnessBoost,
        ),
      );

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
    });
  }

  @override
  Future<List<Map<dynamic, dynamic>>> detectTextFromImage({
    required img.Image image,
    bool includeAllConfidenceScores = false,
    String? debugDumpDir,
    bool trimRecognitionWhitespace = false,
    bool enhanceRecognitionCrops = false,
    double recognitionContrastBoost = 0.08,
    double recognitionBrightnessBoost = 0.02,
  }) async {
    return _runExclusive(() async {
      await _ensureInitializedWithOptions(
        debugDumpDir: debugDumpDir,
        processingOptions: _buildProcessingOptions(
          trimRecognitionWhitespace: trimRecognitionWhitespace,
          enhanceRecognitionCrops: enhanceRecognitionCrops,
          recognitionContrastBoost: recognitionContrastBoost,
          recognitionBrightnessBoost: recognitionBrightnessBoost,
        ),
      );

      final result = await _processor!.processImage(
        image,
        includeAllConfidenceScores: includeAllConfidenceScores,
      );

      return _convertResultToMap(result);
    });
  }

  @override
  Future<bool> hasText({required String imagePath}) async {
    return _runExclusive(() async {
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
    });
  }

  @override
  Future<bool> hasTextInImage({required img.Image image}) async {
    return _runExclusive(() async {
      await _ensureInitialized();

      final result = await _processor!.hasHighConfidenceText(image);
      return result.hasText;
    });
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
        'textOrientation': result.textOrientations[i],
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
    unawaited(_disposeProcessor());
    _processor = null;
    _modelPath = null;
    _initializationFuture = null;
  }
}

class _ProcessorConfiguration {
  const _ProcessorConfiguration({
    required this.debugDumpDir,
    required this.processingOptions,
  });

  final String? debugDumpDir;
  final OcrProcessingOptions processingOptions;

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is _ProcessorConfiguration &&
          debugDumpDir == other.debugDumpDir &&
          processingOptions == other.processingOptions;

  @override
  int get hashCode => Object.hash(debugDumpDir, processingOptions);
}
