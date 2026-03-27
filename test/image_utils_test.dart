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
      ]);

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
