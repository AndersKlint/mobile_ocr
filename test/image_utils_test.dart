import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as img;
import 'package:mobile_ocr/src/ocr/image_utils.dart';
import 'package:mobile_ocr/src/ocr/text_detector.dart';
import 'package:mobile_ocr/src/ocr/types.dart';

void main() {
  group('ImageUtils', () {
    test('orderPointsClockwise matches Paddle ordering', () {
      final points = [
        const Point(90, 15),
        const Point(12, 18),
        const Point(95, 62),
        const Point(8, 65),
      ];

      final ordered = ImageUtils.orderPointsClockwise(points);

      expect(ordered, const [
        Point(12, 18),
        Point(90, 15),
        Point(95, 62),
        Point(8, 65),
      ]);
    });

    test('cropTextRegion rectifies quadrilateral instead of bbox crop', () {
      final source = img.Image(width: 200, height: 120);
      img.fill(source, color: img.ColorRgb8(255, 255, 255));

      for (int y = 20; y < 95; y++) {
        for (int x = 30; x < 175; x++) {
          source.setPixelRgba(x, y, 0, 0, 0, 255);
        }
      }

      final crop = ImageUtils.cropTextRegion(source, const [
        Point(30, 30),
        Point(170, 20),
        Point(175, 88),
        Point(25, 94),
      ], trimWhitespace: true);

      expect(crop.width, closeTo(150, 2));
      expect(crop.height, inInclusiveRange(48, 68));

      final center = crop.getPixel(crop.width ~/ 2, crop.height ~/ 2);
      expect(center.r, lessThan(30));
      expect(center.g, lessThan(30));
      expect(center.b, lessThan(30));

      final topMiddle = crop.getPixel(crop.width ~/ 2, 1);
      expect(topMiddle.r, lessThan(30));
      expect(topMiddle.g, lessThan(30));
      expect(topMiddle.b, lessThan(30));
    });

    test('cropTextRegion does not trim vertical whitespace by default', () {
      final source = img.Image(width: 180, height: 100);
      img.fill(source, color: img.ColorRgb8(255, 255, 255));

      for (int y = 35; y < 65; y++) {
        for (int x = 20; x < 160; x++) {
          source.setPixelRgba(x, y, 0, 0, 0, 255);
        }
      }

      final crop = ImageUtils.cropTextRegion(source, const [
        Point(20, 0),
        Point(160, 0),
        Point(160, 99),
        Point(20, 99),
      ]);

      expect(crop.height, 99);
    });

    test('cropTextRegion trims vertical whitespace when enabled', () {
      final source = img.Image(width: 180, height: 100);
      img.fill(source, color: img.ColorRgb8(255, 255, 255));

      for (int y = 35; y < 65; y++) {
        for (int x = 20; x < 160; x++) {
          source.setPixelRgba(x, y, 0, 0, 0, 255);
        }
      }

      final crop = ImageUtils.cropTextRegion(source, const [
        Point(20, 0),
        Point(160, 0),
        Point(160, 99),
        Point(20, 99),
      ], trimWhitespace: true);

      expect(crop.height, lessThan(45));
      expect(crop.height, greaterThan(25));
    });

    test('rotateOrthogonal rotates 90 degrees clockwise', () {
      final source = img.Image(width: 2, height: 3, numChannels: 4);
      source.setPixelRgba(0, 0, 10, 0, 0, 255);
      source.setPixelRgba(1, 0, 20, 0, 0, 255);
      source.setPixelRgba(0, 1, 30, 0, 0, 255);
      source.setPixelRgba(1, 1, 40, 0, 0, 255);
      source.setPixelRgba(0, 2, 50, 0, 0, 255);
      source.setPixelRgba(1, 2, 60, 0, 0, 255);

      final rotated = ImageUtils.rotateOrthogonal(source, angle: 90);

      expect(rotated.width, 3);
      expect(rotated.height, 2);
      expect(rotated.getPixel(0, 0).r, 50);
      expect(rotated.getPixel(1, 0).r, 30);
      expect(rotated.getPixel(2, 0).r, 10);
      expect(rotated.getPixel(0, 1).r, 60);
      expect(rotated.getPixel(1, 1).r, 40);
      expect(rotated.getPixel(2, 1).r, 20);
    });

    test('rotateOrthogonal rotates 180 degrees', () {
      final source = img.Image(width: 2, height: 2, numChannels: 4);
      source.setPixelRgba(0, 0, 10, 0, 0, 255);
      source.setPixelRgba(1, 0, 20, 0, 0, 255);
      source.setPixelRgba(0, 1, 30, 0, 0, 255);
      source.setPixelRgba(1, 1, 40, 0, 0, 255);

      final rotated = ImageUtils.rotateOrthogonal(source, angle: 180);

      expect(rotated.width, 2);
      expect(rotated.height, 2);
      expect(rotated.getPixel(0, 0).r, 40);
      expect(rotated.getPixel(1, 0).r, 30);
      expect(rotated.getPixel(0, 1).r, 20);
      expect(rotated.getPixel(1, 1).r, 10);
    });

    test('rotateOrthogonal rotates 270 degrees clockwise', () {
      final source = img.Image(width: 2, height: 3, numChannels: 4);
      source.setPixelRgba(0, 0, 10, 0, 0, 255);
      source.setPixelRgba(1, 0, 20, 0, 0, 255);
      source.setPixelRgba(0, 1, 30, 0, 0, 255);
      source.setPixelRgba(1, 1, 40, 0, 0, 255);
      source.setPixelRgba(0, 2, 50, 0, 0, 255);
      source.setPixelRgba(1, 2, 60, 0, 0, 255);

      final rotated = ImageUtils.rotateOrthogonal(source, angle: 270);

      expect(rotated.width, 3);
      expect(rotated.height, 2);
      expect(rotated.getPixel(0, 0).r, 20);
      expect(rotated.getPixel(1, 0).r, 40);
      expect(rotated.getPixel(2, 0).r, 60);
      expect(rotated.getPixel(0, 1).r, 10);
      expect(rotated.getPixel(1, 1).r, 30);
      expect(rotated.getPixel(2, 1).r, 50);
    });

    test(
      'cropTextRegion defaults ambiguous boxes to landscape when preferred',
      () {
        final source = img.Image(width: 120, height: 120);
        img.fill(source, color: img.ColorRgb8(255, 255, 255));

        final crop = ImageUtils.cropTextRegion(source, const [
          Point(30, 20),
          Point(70, 20),
          Point(70, 70),
          Point(30, 70),
        ], preferLandscape: true);

        expect(crop.width, 50);
        expect(crop.height, 40);
      },
    );

    test(
      'cropTextRegion rotates ambiguous wider boxes when landscape is preferred',
      () {
        final source = img.Image(width: 120, height: 120);
        img.fill(source, color: img.ColorRgb8(255, 255, 255));

        final crop = ImageUtils.cropTextRegion(source, const [
          Point(20, 30),
          Point(70, 30),
          Point(70, 70),
          Point(20, 70),
        ], preferLandscape: true);

        expect(crop.width, 40);
        expect(crop.height, 50);
      },
    );

    test('shouldRotateRecognitionRegion keeps tall boxes portrait', () {
      expect(
        ImageUtils.shouldRotateRecognitionRegion(
          cropWidth: 40,
          cropHeight: 80,
          preferLandscape: true,
        ),
        isTrue,
      );
    });

    test('expandBox grows long horizontal quads symmetrically', () {
      const box = [
        Point(100, 200),
        Point(1100, 200),
        Point(1100, 300),
        Point(100, 300),
      ];

      final expanded = ImageUtils.expandBox(
        box,
        horizontalPaddingRatio: 0.05,
        verticalPaddingRatio: 0.5,
        imageWidth: 2000,
        imageHeight: 1000,
      );

      expect(expanded[0].x, closeTo(50, 0.01));
      expect(expanded[1].x, closeTo(1150, 0.01));
      expect(expanded[0].y, closeTo(150, 0.01));
      expect(expanded[2].y, closeTo(350, 0.01));
    });

    test('trimVerticalWhitespace removes large top and bottom margins', () {
      final source = img.Image(width: 180, height: 100);
      img.fill(source, color: img.ColorRgb8(255, 255, 255));

      for (int y = 35; y < 65; y++) {
        for (int x = 20; x < 160; x++) {
          source.setPixelRgba(x, y, 0, 0, 0, 255);
        }
      }

      final trimmed = ImageUtils.trimVerticalWhitespace(source);

      expect(trimmed.width, 180);
      expect(trimmed.height, lessThan(45));
      expect(trimmed.height, greaterThan(25));

      final center = trimmed.getPixel(trimmed.width ~/ 2, trimmed.height ~/ 2);
      expect(center.r, lessThan(30));
      expect(center.g, lessThan(30));
      expect(center.b, lessThan(30));
    });

    test('trimVerticalWhitespace supports light text on dark background', () {
      final source = img.Image(width: 180, height: 100);
      img.fill(source, color: img.ColorRgb8(0, 0, 0));

      for (int y = 38; y < 62; y++) {
        for (int x = 20; x < 160; x++) {
          source.setPixelRgba(x, y, 255, 255, 255, 255);
        }
      }

      final trimmed = ImageUtils.trimVerticalWhitespace(source);

      expect(trimmed.width, 180);
      expect(trimmed.height, lessThan(40));
      expect(trimmed.height, greaterThan(20));

      final center = trimmed.getPixel(trimmed.width ~/ 2, trimmed.height ~/ 2);
      expect(center.r, greaterThan(225));
      expect(center.g, greaterThan(225));
      expect(center.b, greaterThan(225));
    });

    test(
      'trimVerticalWhitespace ignores weak detached band below main line',
      () {
        final source = img.Image(width: 240, height: 140);
        img.fill(source, color: img.ColorRgb8(255, 255, 255));

        for (int y = 24; y < 84; y++) {
          for (int x = 20; x < 220; x++) {
            source.setPixelRgba(x, y, 0, 0, 0, 255);
          }
        }

        for (int y = 118; y < 126; y++) {
          for (int x = 70; x < 170; x++) {
            source.setPixelRgba(x, y, 0, 0, 0, 255);
          }
        }

        final trimmed = ImageUtils.trimVerticalWhitespace(source);

        expect(trimmed.height, lessThan(80));
        expect(trimmed.height, greaterThan(55));

        final bottomCenter = trimmed.getPixel(
          trimmed.width ~/ 2,
          trimmed.height - 2,
        );
        expect(bottomCenter.r, greaterThan(225));
        expect(bottomCenter.g, greaterThan(225));
        expect(bottomCenter.b, greaterThan(225));
      },
    );

    test('enhanceRecognitionCrop boosts contrast and brightness mildly', () {
      final source = img.Image(width: 3, height: 1);
      source.setPixelRgba(0, 0, 0, 0, 0, 255);
      source.setPixelRgba(1, 0, 170, 170, 170, 255);
      source.setPixelRgba(2, 0, 240, 240, 240, 255);

      final enhanced = ImageUtils.enhanceRecognitionCrop(source);

      final black = enhanced.getPixel(0, 0);
      expect(black.r, 0);
      expect(black.g, 0);
      expect(black.b, 0);

      final mid = enhanced.getPixel(1, 0);
      expect(mid.r, greaterThan(170));
      expect(mid.g, greaterThan(170));
      expect(mid.b, greaterThan(170));

      final nearWhite = enhanced.getPixel(2, 0);
      expect(nearWhite.r, greaterThanOrEqualTo(250));
      expect(nearWhite.g, greaterThanOrEqualTo(250));
      expect(nearWhite.b, greaterThanOrEqualTo(250));
    });

    test('enhanceRecognitionCrop returns original image when disabled', () {
      final source = img.Image(width: 2, height: 1);
      source.setPixelRgba(0, 0, 10, 20, 30, 255);
      source.setPixelRgba(1, 0, 40, 50, 60, 255);

      final enhanced = ImageUtils.enhanceRecognitionCrop(
        source,
        contrastBoost: 0,
        brightnessBoost: 0,
      );

      expect(identical(enhanced, source), isTrue);
    });
  });

  group('TextDetector resize', () {
    test('calculateResizeDimensions matches Paddle max-side behavior', () {
      expect(TextDetector.calculateResizeDimensions(4032, 3024), (960, 704));
      expect(TextDetector.calculateResizeDimensions(640, 480), (640, 480));
      expect(TextDetector.calculateResizeDimensions(20, 20), (32, 32));
    });

    test('unclipBox expands rectangle like DB postprocess', () {
      final box = const [
        Point(100, 200),
        Point(1100, 200),
        Point(1100, 300),
        Point(100, 300),
      ];

      final expanded = TextDetector.unclipBox(box, 1.5);

      expect(expanded.length, greaterThanOrEqualTo(4));
      final minX = expanded.map((p) => p.x).reduce((a, b) => a < b ? a : b);
      final maxX = expanded.map((p) => p.x).reduce((a, b) => a > b ? a : b);
      final minY = expanded.map((p) => p.y).reduce((a, b) => a < b ? a : b);
      final maxY = expanded.map((p) => p.y).reduce((a, b) => a > b ? a : b);

      expect(minX, lessThan(100));
      expect(maxX, greaterThan(1100));
      expect(minY, lessThan(200));
      expect(maxY, greaterThan(300));
    });
  });
}
