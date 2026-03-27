import 'dart:io';

import 'package:mobile_ocr/models/text_block.dart';

import 'mobile_ocr_plugin_platform_interface.dart';

class MobileOcr {
  Future<String?> getPlatformVersion() {
    return MobileOcrPlatform.instance.getPlatformVersion();
  }

  Future<ModelPreparationStatus> prepareModels() async {
    final result = await MobileOcrPlatform.instance.prepareModels();
    return ModelPreparationStatus.fromMap(result);
  }

  /// Detect text in an image at the provided file system path.
  ///
  /// [imagePath] must point to a readable PNG or JPEG file.
  ///
  /// By default, only returns results with confidence >= 0.8. Set
  /// [includeAllConfidenceScores] to true to include detections down to 0.5.
  ///
  /// [debugDumpDir] enables debug output: source image, detector input/output,
  /// cropped text regions, and recognition model inputs are saved to this
  /// directory for troubleshooting OCR issues. All intermediate images are
  /// dumped (no limit on count).
  ///
  /// [trimRecognitionWhitespace] enables post-crop vertical whitespace
  /// trimming. This can help when detected boxes are loose, but adds
  /// processing time and may strip detached punctuation. Disabled by
  /// default to match PaddleOCR behavior.
  ///
  /// [enhanceRecognitionCrops] applies mild contrast/brightness boost to
  /// crops before recognition. Can help with faint text but may hurt
  /// already-clear crops. Disabled by default.
  ///
  /// [recognitionContrastBoost] and [recognitionBrightnessBoost] control
  /// the intensity of the enhancement when enabled. Defaults are 0.08 and
  /// 0.02 respectively (mild). Higher values increase contrast and brighten
  /// the image more aggressively.
  Future<List<TextBlock>> detectText({
    required String imagePath,
    bool includeAllConfidenceScores = false,
    String? debugDumpDir,
    bool trimRecognitionWhitespace = false,
    bool enhanceRecognitionCrops = false,
    double recognitionContrastBoost = 0.08,
    double recognitionBrightnessBoost = 0.02,
  }) async {
    final file = File(imagePath);
    if (!file.existsSync()) {
      throw ArgumentError('Image file does not exist at path: $imagePath');
    }

    final results = await MobileOcrPlatform.instance.detectText(
      imagePath: file.path,
      includeAllConfidenceScores: includeAllConfidenceScores,
      debugDumpDir: debugDumpDir,
      trimRecognitionWhitespace: trimRecognitionWhitespace,
      enhanceRecognitionCrops: enhanceRecognitionCrops,
      recognitionContrastBoost: recognitionContrastBoost,
      recognitionBrightnessBoost: recognitionBrightnessBoost,
    );
    return results.map(TextBlock.fromMap).toList(growable: false);
  }

  /// Quickly determine whether the image contains high-confidence text.
  ///
  /// Returns `true` if at least one detected text block has confidence >= 0.9.
  /// Only the detection stage runs, making this faster than full recognition.
  Future<bool> hasText({required String imagePath}) async {
    final file = File(imagePath);
    if (!file.existsSync()) {
      throw ArgumentError('Image file does not exist at path: $imagePath');
    }

    return MobileOcrPlatform.instance.hasText(imagePath: file.path);
  }
}

/// Describes the current preparation status of the native OCR model cache.
class ModelPreparationStatus {
  final bool isReady;
  final String? version;
  final String? modelPath;

  ModelPreparationStatus({required this.isReady, this.version, this.modelPath});

  factory ModelPreparationStatus.fromMap(Map<dynamic, dynamic> map) {
    return ModelPreparationStatus(
      isReady: map['isReady'] == true,
      version: map['version'] as String?,
      modelPath: map['modelPath'] as String?,
    );
  }
}
