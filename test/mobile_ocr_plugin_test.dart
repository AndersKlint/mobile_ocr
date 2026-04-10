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

  test('TextOrientation exposes canonical clockwise rotation helpers', () {
    expect(TextOrientation.portraitUp.clockwiseQuarterTurns, 0);
    expect(TextOrientation.landscapeUp.clockwiseQuarterTurns, 1);
    expect(TextOrientation.portraitDown.clockwiseQuarterTurns, 2);
    expect(TextOrientation.landscapeDown.clockwiseQuarterTurns, 3);

    expect(TextOrientation.portraitUp.clockwiseDegrees, 0);
    expect(TextOrientation.landscapeUp.clockwiseDegrees, 90);
    expect(TextOrientation.portraitDown.clockwiseDegrees, 180);
    expect(TextOrientation.landscapeDown.clockwiseDegrees, 270);
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
      'textOrientation': 'landscapeUp',
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
      'textOrientation': 'landscapeUp',
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
      'textOrientation': 'landscapeDown',
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

  test('TextBlock reads textOrientation from map', () {
    final block = TextBlock.fromMap({
      'text': 'hello',
      'confidence': 0.9,
      'textOrientation': 'portraitDown',
      'points': [
        {'x': 1.0, 'y': 2.0},
        {'x': 5.0, 'y': 2.0},
        {'x': 5.0, 'y': 6.0},
        {'x': 1.0, 'y': 6.0},
      ],
    });

    expect(block.textOrientation, TextOrientation.portraitDown);
  });

  test('TextBlock defaults isRotated180 to false when omitted', () {
    final block = TextBlock.fromMap({
      'text': 'hello',
      'confidence': 0.9,
      'textOrientation': 'landscapeUp',
      'points': [
        {'x': 1.0, 'y': 2.0},
        {'x': 5.0, 'y': 2.0},
      ],
    });

    expect(block.isRotated180, isFalse);
  });

  test('TextBlock requires textOrientation in map', () {
    expect(
      () => TextBlock.fromMap({
        'text': 'hello',
        'confidence': 0.9,
        'points': [
          {'x': 1.0, 'y': 2.0},
          {'x': 5.0, 'y': 2.0},
          {'x': 5.0, 'y': 6.0},
          {'x': 1.0, 'y': 6.0},
        ],
      }),
      throwsArgumentError,
    );
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
        textOrientations: [TextOrientation.landscapeUp.name],
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
    expect(result.textOrientations.single, TextOrientation.portraitUp.name);
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
        textOrientations: [TextOrientation.portraitDown.name],
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
    expect(result.textOrientations.single, TextOrientation.portraitUp.name);
    expect(result.isRotated180.single, isTrue);
  });

  test('resolves text orientation from crop normalization state', () {
    final portraitBox = ocr.TextBox(const [
      ocr.Point(10, 10),
      ocr.Point(50, 10),
      ocr.Point(50, 110),
      ocr.Point(10, 110),
    ]);
    final landscapeBox = ocr.TextBox(const [
      ocr.Point(10, 10),
      ocr.Point(110, 10),
      ocr.Point(110, 50),
      ocr.Point(10, 50),
    ]);

    expect(
      OcrProcessor.resolveTextOrientationForTest(
        portraitBox,
        false,
        preferLandscape: true,
      ),
      TextOrientation.landscapeUp.name,
    );
    expect(
      OcrProcessor.resolveTextOrientationForTest(
        portraitBox,
        true,
        preferLandscape: true,
      ),
      TextOrientation.landscapeDown.name,
    );
    expect(
      OcrProcessor.resolveTextOrientationForTest(
        landscapeBox,
        false,
        preferLandscape: false,
      ),
      TextOrientation.portraitUp.name,
    );
    expect(
      OcrProcessor.resolveTextOrientationForTest(
        landscapeBox,
        true,
        preferLandscape: false,
      ),
      TextOrientation.portraitDown.name,
    );
  });

  test('normalizes landscape orientation to canonical 90-degree rotation', () {
    expect(
      OcrProcessor.normalizeOrientationAngleForTest(
        sourceIsLandscape: false,
        detectedIsLandscape: true,
      ),
      90,
    );
    expect(
      OcrProcessor.normalizeOrientationAngleForTest(
        sourceIsLandscape: true,
        detectedIsLandscape: true,
      ),
      0,
    );
  });

  test('defaults unclassified box direction to classified majority', () {
    expect(
      OcrProcessor.resolveDefaultRotated180ForTest(
        classificationMask: const [true, false, true, false, true],
        rotationStates: const [true, false, true, false, false],
      ),
      isTrue,
    );
    expect(
      OcrProcessor.resolveDefaultRotated180ForTest(
        classificationMask: const [true, false, true],
        rotationStates: const [false, true, false],
      ),
      isFalse,
    );
  });

  test('defaults ambiguous recognition boxes to majority landscape', () {
    expect(
      OcrProcessor.resolveRecognitionBoxLandscapePreferenceForTest([
        const [
          ocr.Point(0, 0),
          ocr.Point(120, 0),
          ocr.Point(120, 30),
          ocr.Point(0, 30),
        ],
        const [
          ocr.Point(10, 10),
          ocr.Point(50, 10),
          ocr.Point(50, 60),
          ocr.Point(10, 60),
        ],
        const [
          ocr.Point(0, 40),
          ocr.Point(140, 40),
          ocr.Point(140, 80),
          ocr.Point(0, 80),
        ],
      ], fallback: false),
      isTrue,
    );
  });

  test('keeps page fallback when recognition box majority is undecided', () {
    expect(
      OcrProcessor.resolveRecognitionBoxLandscapePreferenceForTest([
        const [
          ocr.Point(10, 10),
          ocr.Point(50, 10),
          ocr.Point(50, 60),
          ocr.Point(10, 60),
        ],
        const [
          ocr.Point(20, 20),
          ocr.Point(70, 20),
          ocr.Point(70, 70),
          ocr.Point(20, 70),
        ],
      ], fallback: true),
      isTrue,
    );
  });

  test(
    'character boxes default ambiguous boxes to landscape when preferred',
    () {
      final box = ocr.TextBox(const [
        ocr.Point(10, 10),
        ocr.Point(50, 10),
        ocr.Point(50, 60),
        ocr.Point(10, 60),
      ]);

      final characters = OcrProcessor.buildCharacterBoxesForTest(
        box,
        [
          ocr.CharacterSpan(
            text: 'a',
            confidence: 0.9,
            startRatio: 0.0,
            endRatio: 0.5,
          ),
        ],
        false,
        preferLandscape: true,
      );

      expect(characters, hasLength(1));
      expect(characters.single.points, const [
        ocr.Point(10, 35),
        ocr.Point(50, 35),
        ocr.Point(50, 60),
        ocr.Point(10, 60),
      ]);
    },
  );
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
