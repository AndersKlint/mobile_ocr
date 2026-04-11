import '../../models/text_block.dart' show TextOrientation;
import 'image_utils.dart';
import 'types.dart';

class OcrOrientationMapper {
  static OcrResult mapResultToOriginalOrientation(
    OcrResult result, {
    required int angle,
    required int originalWidth,
    required int originalHeight,
  }) {
    return OcrResult(
      boxes: result.boxes
          .map(
            (box) => TextBox(
              ImageUtils.orderPointsClockwise(
                _mapPointsToOriginalOrientation(
                  box.points,
                  angle: angle,
                  originalWidth: originalWidth,
                  originalHeight: originalHeight,
                ),
              ),
            ),
          )
          .toList(growable: false),
      texts: List<String>.from(result.texts),
      scores: List<double>.from(result.scores),
      characters: result.characters
          .map(
            (characters) => characters
                .map(
                  (character) => CharacterBox(
                    text: character.text,
                    confidence: character.confidence,
                    points: ImageUtils.orderPointsClockwise(
                      _mapPointsToOriginalOrientation(
                        character.points,
                        angle: angle,
                        originalWidth: originalWidth,
                        originalHeight: originalHeight,
                      ),
                    ),
                  ),
                )
                .toList(growable: false),
          )
          .toList(growable: false),
      textOrientations: result.textOrientations
          .map(
            (orientation) => rotateTextOrientation(orientation, angle: angle),
          )
          .toList(growable: false),
    );
  }

  static String resolveTextOrientation(
    TextBox textBox,
    bool rotated180, {
    required bool preferLandscape,
  }) {
    final rotateForRecognition = ImageUtils.shouldRotateRecognitionBox(
      ImageUtils.orderPointsClockwise(textBox.points),
      preferLandscape: preferLandscape,
    );
    if (rotateForRecognition) {
      return rotated180
          ? TextOrientation.landscapeDown.name
          : TextOrientation.landscapeUp.name;
    }

    return rotated180
        ? TextOrientation.portraitDown.name
        : TextOrientation.portraitUp.name;
  }

  static String rotateTextOrientation(
    String orientation, {
    required int angle,
  }) {
    final normalizedAngle = ((angle % 360) + 360) % 360;
    final baseAngle = switch (orientation) {
      'portraitUp' => 0,
      'landscapeUp' => 90,
      'portraitDown' => 180,
      'landscapeDown' => 270,
      _ => 0,
    };
    final rotatedAngle = (baseAngle - normalizedAngle + 360) % 360;
    return switch (rotatedAngle) {
      0 => TextOrientation.portraitUp.name,
      90 => TextOrientation.landscapeUp.name,
      180 => TextOrientation.portraitDown.name,
      270 => TextOrientation.landscapeDown.name,
      _ => TextOrientation.portraitUp.name,
    };
  }

  static int normalizeOrientationAngle({
    required bool sourceIsLandscape,
    required bool detectedIsLandscape,
  }) {
    if (sourceIsLandscape == detectedIsLandscape) {
      return 0;
    }
    return 90;
  }

  static bool resolveDefaultRotated180({
    required List<bool> classificationMask,
    required List<bool> rotationStates,
  }) {
    var rotatedCount = 0;
    var uprightCount = 0;

    for (int index = 0; index < classificationMask.length; index++) {
      if (!classificationMask[index]) {
        continue;
      }
      if (rotationStates[index]) {
        rotatedCount++;
      } else {
        uprightCount++;
      }
    }

    return rotatedCount > uprightCount;
  }

  static bool resolveRecognitionBoxLandscapePreference(
    List<List<Point>> boxes, {
    required bool fallback,
  }) {
    var landscapeCount = 0;
    var portraitCount = 0;

    for (final box in boxes) {
      final prefersLandscape =
          ImageUtils.inferRecognitionBoxLandscapePreference(box);
      if (prefersLandscape == null) {
        continue;
      }
      if (prefersLandscape) {
        landscapeCount++;
      } else {
        portraitCount++;
      }
    }

    if (landscapeCount == portraitCount) {
      return fallback;
    }

    return landscapeCount > portraitCount;
  }

  static List<Point> _mapPointsToOriginalOrientation(
    List<Point> points, {
    required int angle,
    required int originalWidth,
    required int originalHeight,
  }) {
    return points
        .map(
          (point) => _mapPointToOriginalOrientation(
            point,
            angle: angle,
            originalWidth: originalWidth,
            originalHeight: originalHeight,
          ),
        )
        .toList(growable: false);
  }

  static Point _mapPointToOriginalOrientation(
    Point point, {
    required int angle,
    required int originalWidth,
    required int originalHeight,
  }) {
    return switch (angle) {
      90 => Point(point.y, (originalHeight - 1) - point.x),
      180 => Point(
        (originalWidth - 1) - point.x,
        (originalHeight - 1) - point.y,
      ),
      270 => Point((originalWidth - 1) - point.y, point.x),
      _ => point,
    };
  }
}
