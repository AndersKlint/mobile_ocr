import 'dart:io';

import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:mobile_ocr/mobile_ocr.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  group('textbook OCR integration tests', () {
    late MobileOcr ocr;

    setUpAll(() async {
      ocr = MobileOcr();
      final result = await ocr.prepareModels();
      expect(result.isReady, isTrue, reason: 'Models should be ready');
    });

    test('full textbook image should find expected text', () async {
      final testAsset = 'assets/test_ocr/textbook.jpg';
      final imagePath = await _loadAssetToTemp(testAsset);

      final result = await ocr.detectText(imagePath: imagePath);

      final allText = result.map((block) => block.text).join('\n');
      expect(
        allText,
        contains('书面语。用强力取得或保持。'),
        reason:
            'Expected text not found in full image.\n'
            'Found text:\n$allText',
      );

      await File(imagePath).parent.delete(recursive: true);
    });

    test('cropped textbook image should find expected text', () async {
      final testAsset = 'assets/test_ocr/textbook._crop.jpg';
      final imagePath = await _loadAssetToTemp(testAsset);

      final result = await ocr.detectText(imagePath: imagePath);

      final allText = result.map((block) => block.text).join('\n');
      expect(
        allText,
        contains('书面语。用强力取得或保持。'),
        reason:
            'Expected text not found in cropped image.\n'
            'Found text:\n$allText',
      );

      await File(imagePath).parent.delete(recursive: true);
    });
  });
}

Future<String> _loadAssetToTemp(String assetPath) async {
  final tempDir = await Directory.systemTemp.createTemp('textbook_test_');
  final fileName = assetPath.split('/').last;
  final imagePath = '${tempDir.path}/$fileName';

  final data = await rootBundle.load(assetPath);
  await File(imagePath).writeAsBytes(
    data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes),
  );

  return imagePath;
}
