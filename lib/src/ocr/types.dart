class Point {
  final double x;
  final double y;

  const Point(this.x, this.y);

  @override
  String toString() => 'Point($x, $y)';

  @override
  bool operator ==(Object other) =>
      identical(this, other) || other is Point && x == other.x && y == other.y;

  @override
  int get hashCode => x.hashCode ^ y.hashCode;

  Point operator +(Point other) => Point(x + other.x, y + other.y);
  Point operator -(Point other) => Point(x - other.x, y - other.y);
  Point operator *(double scalar) => Point(x * scalar, y * scalar);
}

class TextBox {
  final List<Point> points;

  TextBox(this.points);

  Rect boundingRect() {
    if (points.isEmpty) {
      return Rect.zero;
    }

    double minX = points[0].x;
    double maxX = points[0].x;
    double minY = points[0].y;
    double maxY = points[0].y;

    for (final point in points) {
      if (point.x < minX) minX = point.x;
      if (point.x > maxX) maxX = point.x;
      if (point.y < minY) minY = point.y;
      if (point.y > maxY) maxY = point.y;
    }

    return Rect(minX, minY, maxX, maxY);
  }
}

class Rect {
  final double left;
  final double top;
  final double right;
  final double bottom;

  const Rect(this.left, this.top, this.right, this.bottom);

  static const zero = Rect(0, 0, 0, 0);

  double get width => right - left;
  double get height => bottom - top;

  bool get isEmpty => width <= 0 || height <= 0;
}

class CharacterSpan {
  final String text;
  final double confidence;
  final double startRatio;
  final double endRatio;

  CharacterSpan({
    required this.text,
    required this.confidence,
    required this.startRatio,
    required this.endRatio,
  });
}

class CharacterBox {
  final String text;
  final double confidence;
  final List<Point> points;

  CharacterBox({
    required this.text,
    required this.confidence,
    required this.points,
  });
}

class RecognitionResult {
  final String text;
  final double confidence;
  final List<CharacterSpan> characterSpans;

  RecognitionResult({
    required this.text,
    required this.confidence,
    required this.characterSpans,
  });
}

class OcrResult {
  final List<TextBox> boxes;
  final List<String> texts;
  final List<double> scores;
  final List<List<CharacterBox>> characters;

  OcrResult({
    required this.boxes,
    required this.texts,
    required this.scores,
    required this.characters,
  });
}

class DetectionCandidate {
  final TextBox box;
  final double score;

  DetectionCandidate(this.box, this.score);
}

class DetectionStageSummary {
  final int examinedDetections;
  final double? maxDetectionScore;
  final List<DetectionCandidate> candidates;

  DetectionStageSummary({
    required this.examinedDetections,
    required this.maxDetectionScore,
    required this.candidates,
  });
}

class QuickCheckResult {
  final bool hasText;
  final bool detectorHit;
  final int examinedDetections;
  final int candidateCount;
  final int evaluatedCandidates;
  final double? maxDetectionScore;
  final double? bestRecognitionScore;
  final String? bestRecognitionText;
  final double? matchedDetectionScore;

  QuickCheckResult({
    required this.hasText,
    required this.detectorHit,
    required this.examinedDetections,
    required this.candidateCount,
    required this.evaluatedCandidates,
    required this.maxDetectionScore,
    required this.bestRecognitionScore,
    required this.bestRecognitionText,
    required this.matchedDetectionScore,
  });
}
