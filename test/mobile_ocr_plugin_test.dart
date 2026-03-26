import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:mobile_ocr/mobile_ocr_plugin.dart';
import 'package:mobile_ocr/mobile_ocr_plugin_platform_interface.dart';
import 'package:mobile_ocr/mobile_ocr_plugin_dart.dart';
import 'package:mobile_ocr/models/text_block.dart';
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
  }) async {
    return [];
  }

  @override
  Future<bool> hasText({required String imagePath}) async {
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
}

class _VerifyingMobileOcrPlatform extends MockMobileOcrPlatform {
  String? lastImagePath;
  String? lastDebugDumpDir;
  bool response = false;

  @override
  Future<List<Map<dynamic, dynamic>>> detectText({
    required String imagePath,
    bool includeAllConfidenceScores = false,
    String? debugDumpDir,
  }) async {
    lastImagePath = imagePath;
    lastDebugDumpDir = debugDumpDir;
    return [];
  }

  @override
  Future<bool> hasText({required String imagePath}) async {
    lastImagePath = imagePath;
    return response;
  }
}
