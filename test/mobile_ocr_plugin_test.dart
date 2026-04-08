import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as img;
import 'package:mobile_ocr/mobile_ocr_plugin.dart';
import 'package:mobile_ocr/mobile_ocr_plugin_platform_interface.dart';
import 'package:mobile_ocr/mobile_ocr_plugin_dart.dart';
import 'package:mobile_ocr/models/text_block.dart';
import 'package:mobile_ocr/src/ocr/image_utils.dart';
import 'package:mobile_ocr/src/ocr/ocr_processor.dart';
import 'package:mobile_ocr/src/ocr/types.dart' as ocr;
import 'package:plugin_platform_interface/plugin_platform_interface.dart';

class MockMobileOcrPlatform
    with MockPlatformInterfaceMixin
    implements MobileOcrPlatform {
  @override
  Future<String?> getPlatformVersion() => Future.value('42');

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
    return [];
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
    return [];
  }

  @override
  Future<bool> hasText({required String imagePath}) async {
    return false;
  }

  @override
  Future<bool> hasTextInImage({required img.Image image}) async {
    return false;
  }

  @override
  Future<Map<dynamic, dynamic>> prepareModels() async {
    return {'isReady': true, 'version': 'test', 'modelPath': '/tmp'};
  }
}

void main() {
  final MobileOcrPlatform initialPlatform = MobileOcrPlatform.instance;

  tearDown(() {
    MobileOcrPlatform.instance = initialPlatform;
  });

  test('Platform default instance is DartMobileOcr', () {
    expect(initialPlatform, isInstanceOf<DartMobileOcr>());
  });

  test('getPlatformVersion', () async {
    MobileOcr mobileOcr = MobileOcr();
    MockMobileOcrPlatform fakePlatform = MockMobileOcrPlatform();
    MobileOcrPlatform.instance = fakePlatform;

    expect(await mobileOcr.getPlatformVersion(), '42');
  });

  test('TextBlock computes bounding box from points', () {
    final block = TextBlock.fromMap({
      'text': 'hello',
      'confidence': 0.9,
      'points': [
        {'x': 1.0, 'y': 2.0},
        {'x': 5.0, 'y': 2.0},
        {'x': 5.0, 'y': 6.0},
        {'x': 1.0, 'y': 6.0},
      ],
    });

    expect(block.boundingBox.left, 1.0);
    expect(block.boundingBox.top, 2.0);
    expect(block.boundingBox.width, 4.0);
    expect(block.boundingBox.height, 4.0);
  });

  test('TextBlock falls back to rectangle fields when points absent', () {
    final block = TextBlock.fromMap({
      'text': 'hello',
      'confidence': 0.9,
      'x': 2.0,
      'y': 3.0,
      'width': 8.0,
      'height': 4.0,
    });

    expect(block.points, hasLength(4));
    expect(block.boundingBox.left, 2.0);
    expect(block.boundingBox.top, 3.0);
    expect(block.boundingBox.right, 10.0);
    expect(block.boundingBox.bottom, 7.0);
  });

  test('TextBlock reads isRotated180 from map', () {
    final block = TextBlock.fromMap({
      'text': 'hello',
      'confidence': 0.9,
      'isRotated180': true,
      'points': [
        {'x': 1.0, 'y': 2.0},
        {'x': 5.0, 'y': 2.0},
        {'x': 5.0, 'y': 6.0},
        {'x': 1.0, 'y': 6.0},
      ],
    });

    expect(block.isRotated180, isTrue);
  });

  test('TextBlock defaults isRotated180 to false when omitted', () {
    final block = TextBlock.fromMap({
      'text': 'hello',
      'confidence': 0.9,
      'points': [
        {'x': 1.0, 'y': 2.0},
        {'x': 5.0, 'y': 2.0},
      ],
    });

    expect(block.isRotated180, isFalse);
  });

  test('hasText validates image path exists', () async {
    final mobileOcr = MobileOcr();
    expect(
      () => mobileOcr.hasText(imagePath: '/tmp/does_not_exist.png'),
      throwsArgumentError,
    );
  });

  test('hasText delegates to platform implementation', () async {
    final tempDir = await Directory.systemTemp.createTemp('mobile_ocr_test');
    final tempFile = File('${tempDir.path}/image.png');
    await tempFile.writeAsBytes([0x00]);

    final mobileOcr = MobileOcr();
    final verifyingPlatform = _VerifyingMobileOcrPlatform();
    verifyingPlatform.response = true;
    MobileOcrPlatform.instance = verifyingPlatform;

    final result = await mobileOcr.hasText(imagePath: tempFile.path);
    expect(result, isTrue);
    expect(verifyingPlatform.lastImagePath, tempFile.path);

    await tempDir.delete(recursive: true);
  });

  test('detectText forwards debug dump directory', () async {
    final tempDir = await Directory.systemTemp.createTemp('mobile_ocr_test');
    final tempFile = File('${tempDir.path}/image.png');
    await tempFile.writeAsBytes([0x00]);

    final mobileOcr = MobileOcr();
    final verifyingPlatform = _VerifyingMobileOcrPlatform();
    MobileOcrPlatform.instance = verifyingPlatform;

    await mobileOcr.detectText(
      imagePath: tempFile.path,
      debugDumpDir: '${tempDir.path}/debug',
    );

    expect(verifyingPlatform.lastImagePath, tempFile.path);
    expect(verifyingPlatform.lastDebugDumpDir, '${tempDir.path}/debug');

    await tempDir.delete(recursive: true);
  });

  test('detectText forwards crop processing options', () async {
    final tempDir = await Directory.systemTemp.createTemp('mobile_ocr_test');
    final tempFile = File('${tempDir.path}/image.png');
    await tempFile.writeAsBytes([0x00]);

    final mobileOcr = MobileOcr();
    final verifyingPlatform = _VerifyingMobileOcrPlatform();
    MobileOcrPlatform.instance = verifyingPlatform;

    await mobileOcr.detectText(
      imagePath: tempFile.path,
      trimRecognitionWhitespace: true,
      enhanceRecognitionCrops: true,
      recognitionContrastBoost: 0.04,
      recognitionBrightnessBoost: 0.01,
    );

    expect(verifyingPlatform.lastTrimRecognitionWhitespace, isTrue);
    expect(verifyingPlatform.lastEnhanceRecognitionCrops, isTrue);
    expect(verifyingPlatform.lastRecognitionContrastBoost, 0.04);
    expect(verifyingPlatform.lastRecognitionBrightnessBoost, 0.01);

    await tempDir.delete(recursive: true);
  });

  test('maps rotated boxes back to original orientation', () {
    const originalWidth = 200;
    const originalHeight = 100;
    final originalPoints = <ocr.Point>[
      const ocr.Point(20, 10),
      const ocr.Point(80, 10),
      const ocr.Point(80, 30),
      const ocr.Point(20, 30),
    ];

    final rotatedPoints = ImageUtils.orderPointsClockwise(
      originalPoints
          .map((point) => ocr.Point((originalHeight - 1) - point.y, point.x))
          .toList(growable: false),
    );

    final result = OcrProcessor.mapResultToOriginalOrientationForTest(
      ocr.OcrResult(
        boxes: [ocr.TextBox(rotatedPoints)],
        texts: ['test'],
        scores: [0.9],
        characters: [
          [ocr.CharacterBox(text: 't', confidence: 0.8, points: rotatedPoints)],
        ],
        isRotated180: [false],
      ),
      angle: 90,
      originalWidth: originalWidth,
      originalHeight: originalHeight,
    );

    expect(result.boxes, hasLength(1));
    expect(result.characters.single, hasLength(1));
    expect(result.boxes.single.points, originalPoints);
    expect(result.characters.single.single.points, originalPoints);
  });

  test('maps 180-rotated boxes back to original orientation', () {
    const originalWidth = 200;
    const originalHeight = 100;
    final originalPoints = <ocr.Point>[
      const ocr.Point(20, 10),
      const ocr.Point(80, 10),
      const ocr.Point(80, 30),
      const ocr.Point(20, 30),
    ];

    final rotatedPoints = ImageUtils.orderPointsClockwise(
      originalPoints
          .map(
            (point) => ocr.Point(
              (originalWidth - 1) - point.x,
              (originalHeight - 1) - point.y,
            ),
          )
          .toList(growable: false),
    );

    final result = OcrProcessor.mapResultToOriginalOrientationForTest(
      ocr.OcrResult(
        boxes: [ocr.TextBox(rotatedPoints)],
        texts: ['test'],
        scores: [0.9],
        characters: [
          [ocr.CharacterBox(text: 't', confidence: 0.8, points: rotatedPoints)],
        ],
        isRotated180: [true],
      ),
      angle: 180,
      originalWidth: originalWidth,
      originalHeight: originalHeight,
    );

    expect(result.boxes, hasLength(1));
    expect(result.characters.single, hasLength(1));
    expect(result.boxes.single.points, originalPoints);
    expect(result.characters.single.single.points, originalPoints);
    expect(result.isRotated180.single, isTrue);
  });
}

class _VerifyingMobileOcrPlatform extends MockMobileOcrPlatform {
  String? lastImagePath;
  String? lastDebugDumpDir;
  bool? lastTrimRecognitionWhitespace;
  bool? lastEnhanceRecognitionCrops;
  double? lastRecognitionContrastBoost;
  double? lastRecognitionBrightnessBoost;
  bool response = false;

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
    lastImagePath = imagePath;
    lastDebugDumpDir = debugDumpDir;
    lastTrimRecognitionWhitespace = trimRecognitionWhitespace;
    lastEnhanceRecognitionCrops = enhanceRecognitionCrops;
    lastRecognitionContrastBoost = recognitionContrastBoost;
    lastRecognitionBrightnessBoost = recognitionBrightnessBoost;
    return [];
  }

  @override
  Future<bool> hasText({required String imagePath}) async {
    lastImagePath = imagePath;
    return response;
  }
}
