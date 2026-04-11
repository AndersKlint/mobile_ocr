import 'image_utils.dart';
import 'types.dart';

class OcrCharacterBoxBuilder {
  static List<CharacterBox> build(
    TextBox textBox,
    List<CharacterSpan> spans,
    bool rotated, {
    bool preferLandscape = false,
  }) {
    if (spans.isEmpty) {
      return [];
    }

    final ordered = ImageUtils.orderPointsClockwise(textBox.points);
    if (ordered.length != 4) {
      return [];
    }

    final topLeft = ordered[0];
    final topRight = ordered[1];
    final bottomRight = ordered[2];
    final bottomLeft = ordered[3];
    final rotateForRecognition = ImageUtils.shouldRotateRecognitionBox(
      ordered,
      preferLandscape: preferLandscape,
    );

    const epsilon = 1e-4;

    return spans
        .map((span) {
          var start = span.startRatio;
          var end = span.endRatio;

          if (rotated) {
            final reversedStart = 1.0 - end;
            final reversedEnd = 1.0 - start;
            start = reversedStart.clamp(0.0, 1.0);
            end = reversedEnd.clamp(start + epsilon, 1.0);
          }

          final clampedStart = start.clamp(0.0, 1.0);
          final clampedEnd = end.clamp(clampedStart + epsilon, 1.0);
          if (clampedEnd - clampedStart <= epsilon) {
            return null;
          }

          final points = rotateForRecognition
              ? [
                  _interpolate(bottomLeft, topLeft, clampedStart),
                  _interpolate(bottomLeft, topLeft, clampedEnd),
                  _interpolate(bottomRight, topRight, clampedEnd),
                  _interpolate(bottomRight, topRight, clampedStart),
                ]
              : [
                  _interpolate(topLeft, topRight, clampedStart),
                  _interpolate(topLeft, topRight, clampedEnd),
                  _interpolate(bottomLeft, bottomRight, clampedEnd),
                  _interpolate(bottomLeft, bottomRight, clampedStart),
                ];

          return CharacterBox(
            text: span.text,
            confidence: span.confidence,
            points: ImageUtils.orderPointsClockwise(points),
          );
        })
        .whereType<CharacterBox>()
        .toList(growable: false);
  }

  static Point _interpolate(Point start, Point end, double ratio) {
    final clamped = ratio.clamp(0.0, 1.0);
    return Point(
      start.x + (end.x - start.x) * clamped,
      start.y + (end.y - start.y) * clamped,
    );
  }
}
