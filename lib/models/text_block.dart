import 'dart:ui';

/// Clockwise orientation of recognized text in original image coordinates.
///
/// The naming follows the common mobile/device orientation basis where upright
/// horizontal text is [TextOrientation.portraitUp] and additional orientations
/// advance clockwise from there:
///
/// - [TextOrientation.portraitUp] = `0°`
/// - [TextOrientation.landscapeUp] = `90°`
/// - [TextOrientation.portraitDown] = `180°`
/// - [TextOrientation.landscapeDown] = `270°`
enum TextOrientation { landscapeUp, landscapeDown, portraitUp, portraitDown }

/// Helpers for rendering [TextOrientation] consistently across consumers.
extension TextOrientationRotation on TextOrientation {
  /// The clockwise rotation needed to render text in its OCR reading direction.
  int get clockwiseQuarterTurns => switch (this) {
    TextOrientation.portraitUp => 0,
    TextOrientation.landscapeUp => 1,
    TextOrientation.portraitDown => 2,
    TextOrientation.landscapeDown => 3,
  };

  /// The clockwise rotation needed to render text in its OCR reading direction.
  int get clockwiseDegrees => clockwiseQuarterTurns * 90;
}

/// Represents a detected block of text with its polygon outline.
class TextBlock {
  final String text;
  final double confidence;
  final List<Offset> points;
  final List<CharacterBox> characters;
  final TextOrientation textOrientation;
  final bool isRotated180;

  const TextBlock({
    required this.text,
    required this.confidence,
    required this.points,
    required this.characters,
    this.textOrientation = TextOrientation.portraitUp,
    this.isRotated180 = false,
  });

  Rect get boundingBox {
    if (points.isEmpty) {
      return Rect.zero;
    }

    var minX = points.first.dx;
    var maxX = points.first.dx;
    var minY = points.first.dy;
    var maxY = points.first.dy;

    for (final point in points) {
      if (point.dx < minX) minX = point.dx;
      if (point.dx > maxX) maxX = point.dx;
      if (point.dy < minY) minY = point.dy;
      if (point.dy > maxY) maxY = point.dy;
    }

    return Rect.fromLTRB(minX, minY, maxX, maxY);
  }

  Offset get center => boundingBox.center;

  factory TextBlock.fromMap(Map<dynamic, dynamic> map) {
    final confidence = (map['confidence'] as num?)?.toDouble() ?? 0.0;
    final pointsList = map['points'] as List?;
    final points = (pointsList == null || pointsList.isEmpty)
        ? _fallbackPointsFromRect(map)
        : pointsList
              .whereType<Map<dynamic, dynamic>>()
              .map(
                (point) => Offset(
                  (point['x'] as num).toDouble(),
                  (point['y'] as num).toDouble(),
                ),
              )
              .toList(growable: false);

    final charactersList = map['characters'] as List?;
    final characters = charactersList == null
        ? const <CharacterBox>[]
        : charactersList
              .whereType<Map<dynamic, dynamic>>()
              .map(CharacterBox.fromMap)
              .toList(growable: false);

    return TextBlock(
      text: map['text'] as String? ?? '',
      confidence: confidence,
      points: points,
      characters: characters,
      textOrientation: _parseTextOrientation(map),
      isRotated180: map['isRotated180'] == true,
    );
  }

  Map<String, dynamic> toMap() => {
    'text': text,
    'confidence': confidence,
    'textOrientation': textOrientation.name,
    'isRotated180': isRotated180,
    'points': points
        .map((point) => {'x': point.dx, 'y': point.dy})
        .toList(growable: false),
    'characters': characters
        .map((character) => character.toMap())
        .toList(growable: false),
  };

  static TextOrientation _parseTextOrientation(Map<dynamic, dynamic> map) {
    final rawOrientation = map['textOrientation'];
    if (rawOrientation is String) {
      for (final value in TextOrientation.values) {
        if (value.name == rawOrientation) {
          return value;
        }
      }
    }

    throw ArgumentError('TextBlock map is missing valid textOrientation.');
  }

  static List<Offset> _fallbackPointsFromRect(Map<dynamic, dynamic> map) {
    final x = map['x'] as num?;
    final y = map['y'] as num?;
    final width = map['width'] as num?;
    final height = map['height'] as num?;

    if (x == null || y == null || width == null || height == null) {
      throw ArgumentError(
        'TextBlock map is missing polygon points and fallback rectangle.',
      );
    }

    final left = x.toDouble();
    final top = y.toDouble();
    final blockWidth = width.toDouble();
    final blockHeight = height.toDouble();

    return <Offset>[
      Offset(left, top),
      Offset(left + blockWidth, top),
      Offset(left + blockWidth, top + blockHeight),
      Offset(left, top + blockHeight),
    ];
  }
}

class CharacterBox {
  final String text;
  final double confidence;
  final List<Offset> points;

  const CharacterBox({
    required this.text,
    required this.confidence,
    required this.points,
  });

  Rect get boundingBox {
    if (points.isEmpty) {
      return Rect.zero;
    }

    var minX = points.first.dx;
    var maxX = points.first.dx;
    var minY = points.first.dy;
    var maxY = points.first.dy;

    for (final point in points) {
      if (point.dx < minX) minX = point.dx;
      if (point.dx > maxX) maxX = point.dx;
      if (point.dy < minY) minY = point.dy;
      if (point.dy > maxY) maxY = point.dy;
    }

    return Rect.fromLTRB(minX, minY, maxX, maxY);
  }

  factory CharacterBox.fromMap(Map<dynamic, dynamic> map) {
    final pointsList = map['points'] as List? ?? const [];
    final parsedPoints = pointsList
        .whereType<Map<dynamic, dynamic>>()
        .map(
          (point) => Offset(
            (point['x'] as num).toDouble(),
            (point['y'] as num).toDouble(),
          ),
        )
        .toList(growable: false);

    return CharacterBox(
      text: (map['text'] as String?) ?? '',
      confidence: (map['confidence'] as num?)?.toDouble() ?? 0.0,
      points: parsedPoints,
    );
  }

  Map<String, dynamic> toMap() => {
    'text': text,
    'confidence': confidence,
    'points': points
        .map((point) => {'x': point.dx, 'y': point.dy})
        .toList(growable: false),
  };
}
