import 'dart:io';

import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:mobile_ocr/mobile_ocr.dart';
import 'package:path_provider/path_provider.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  group('Linux OCR Integration Tests', () {
    late MobileOcr ocr;

    setUpAll(() async {
      ocr = MobileOcr();

      final appDir = await getApplicationSupportDirectory();
      final modelsDir = Directory('${appDir.path}/assets/mobile_ocr');

      if (!await modelsDir.exists()) {
        await modelsDir.create(recursive: true);
      }

      const modelFiles = [
        'det.onnx',
        'rec.onnx',
        'cls.onnx',
        'ppocrv5_dict.txt',
      ];

      for (final modelFile in modelFiles) {
        final targetFile = File('${modelsDir.path}/$modelFile');
        if (!await targetFile.exists()) {
          final data = await rootBundle.load('assets/mobile_ocr/$modelFile');
          await targetFile.writeAsBytes(
            data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes),
            flush: true,
          );
        }
      }
    });

    test('prepareModels returns ready status', () async {
      final result = await ocr.prepareModels();
      expect(result.isReady, isTrue);
    });

    test('detectText finds text in image', () async {
      final testAsset = 'assets/test_ocr/ocr_test.jpeg';

      final tempDir = await Directory.systemTemp.createTemp('ocr_test_');
      final imagePath = '${tempDir.path}/ocr_test.jpeg';

      final data = await rootBundle.load(testAsset);
      await File(imagePath).writeAsBytes(
        data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes),
      );

      final result = await ocr.detectText(imagePath: imagePath);

      expect(result.length, greaterThan(0));
      expect(
        result.any((b) => b.text.isNotEmpty),
        isTrue,
        reason: 'Should recognize some text',
      );

      await tempDir.delete(recursive: true);
    });

    test('hasText returns true for image with text', () async {
      final testAsset = 'assets/test_ocr/ocr_test.jpeg';

      final tempDir = await Directory.systemTemp.createTemp(
        'ocr_hastext_test_',
      );
      final imagePath = '${tempDir.path}/ocr_test.jpeg';

      final data = await rootBundle.load(testAsset);
      await File(imagePath).writeAsBytes(
        data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes),
      );

      final hasText = await ocr.hasText(imagePath: imagePath);
      expect(hasText, isTrue);

      await tempDir.delete(recursive: true);
    });
  });
}
