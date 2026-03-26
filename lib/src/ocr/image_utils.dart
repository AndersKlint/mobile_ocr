import 'dart:math' as math;
import 'dart:typed_data';
import 'package:image/image.dart' as img;
import 'types.dart';

class ImageUtils {
  static List<Point> orderPointsClockwise(List<Point> points) {
    if (points.length != 4) {
      return points;
    }

    final rect = List<Point?>.filled(4, null);
    final sums = points.map((point) => point.x + point.y).toList();

    var minSumIndex = 0;
    var maxSumIndex = 0;
    for (int i = 1; i < points.length; i++) {
      if (sums[i] < sums[minSumIndex]) {
        minSumIndex = i;
      }
      if (sums[i] > sums[maxSumIndex]) {
        maxSumIndex = i;
      }
    }

    rect[0] = points[minSumIndex];
    rect[2] = points[maxSumIndex];

    final remaining = <Point>[];
    for (int i = 0; i < points.length; i++) {
      if (i != minSumIndex && i != maxSumIndex) {
        remaining.add(points[i]);
      }
    }

    if (remaining.length != 2) {
      return points;
    }

    final diff0 = remaining[0].y - remaining[0].x;
    final diff1 = remaining[1].y - remaining[1].x;
    if (diff0 <= diff1) {
      rect[1] = remaining[0];
      rect[3] = remaining[1];
    } else {
      rect[1] = remaining[1];
      rect[3] = remaining[0];
    }

    return rect.whereType<Point>().toList(growable: false);
  }

  static img.Image cropTextRegion(img.Image bitmap, List<Point> points) {
    if (points.length != 4) {
      throw ArgumentError('Expected 4 points for text region');
    }

    final ordered = orderPointsClockwise(
      clipBoxToImageBounds(points, bitmap.width, bitmap.height),
    );
    final cropWidth = math
        .max(distance(ordered[0], ordered[1]), distance(ordered[2], ordered[3]))
        .toInt()
        .clamp(1, 10000);
    final cropHeight = math
        .max(distance(ordered[0], ordered[3]), distance(ordered[1], ordered[2]))
        .toInt()
        .clamp(1, 10000);

    final srcPoints = <double>[
      ordered[0].x,
      ordered[0].y,
      ordered[1].x,
      ordered[1].y,
      ordered[2].x,
      ordered[2].y,
      ordered[3].x,
      ordered[3].y,
    ];
    final dstPoints = <double>[
      0,
      0,
      cropWidth.toDouble(),
      0,
      cropWidth.toDouble(),
      cropHeight.toDouble(),
      0,
      cropHeight.toDouble(),
    ];

    final cropped = perspectiveTransform(
      bitmap,
      srcPoints,
      dstPoints,
      cropWidth,
      cropHeight,
    );

    if (cropHeight / cropWidth >= 1.5) {
      return img.copyRotate(cropped, angle: 90);
    }

    return cropped;
  }

  static double quadWidth(List<Point> points) {
    final ordered = orderPointsClockwise(points);
    if (ordered.length != 4) {
      return 0;
    }

    return math.max(
      distance(ordered[0], ordered[1]),
      distance(ordered[2], ordered[3]),
    );
  }

  static double quadHeight(List<Point> points) {
    final ordered = orderPointsClockwise(points);
    if (ordered.length != 4) {
      return 0;
    }

    return math.max(
      distance(ordered[0], ordered[3]),
      distance(ordered[1], ordered[2]),
    );
  }

  static List<Point> expandBox(
    List<Point> points, {
    required double horizontalPaddingRatio,
    required double verticalPaddingRatio,
    required int imageWidth,
    required int imageHeight,
  }) {
    final ordered = orderPointsClockwise(points);
    if (ordered.length != 4) {
      return clipBoxToImageBounds(points, imageWidth, imageHeight);
    }

    final width = quadWidth(ordered);
    final height = quadHeight(ordered);
    if (width <= 0 || height <= 0) {
      return clipBoxToImageBounds(ordered, imageWidth, imageHeight);
    }

    final horizontal = Point(
      ((ordered[1].x - ordered[0].x) + (ordered[2].x - ordered[3].x)) / 2,
      ((ordered[1].y - ordered[0].y) + (ordered[2].y - ordered[3].y)) / 2,
    );
    final vertical = Point(
      ((ordered[3].x - ordered[0].x) + (ordered[2].x - ordered[1].x)) / 2,
      ((ordered[3].y - ordered[0].y) + (ordered[2].y - ordered[1].y)) / 2,
    );

    final horizontalDirection = normalize(horizontal) ?? const Point(1.0, 0.0);
    final verticalDirection = normalize(vertical) ?? const Point(0.0, 1.0);

    final horizontalPadding = width * horizontalPaddingRatio;
    final verticalPadding = height * verticalPaddingRatio;

    final expanded = [
      Point(
        ordered[0].x -
            horizontalDirection.x * horizontalPadding -
            verticalDirection.x * verticalPadding,
        ordered[0].y -
            horizontalDirection.y * horizontalPadding -
            verticalDirection.y * verticalPadding,
      ),
      Point(
        ordered[1].x +
            horizontalDirection.x * horizontalPadding -
            verticalDirection.x * verticalPadding,
        ordered[1].y +
            horizontalDirection.y * horizontalPadding -
            verticalDirection.y * verticalPadding,
      ),
      Point(
        ordered[2].x +
            horizontalDirection.x * horizontalPadding +
            verticalDirection.x * verticalPadding,
        ordered[2].y +
            horizontalDirection.y * horizontalPadding +
            verticalDirection.y * verticalPadding,
      ),
      Point(
        ordered[3].x -
            horizontalDirection.x * horizontalPadding +
            verticalDirection.x * verticalPadding,
        ordered[3].y -
            horizontalDirection.y * horizontalPadding +
            verticalDirection.y * verticalPadding,
      ),
    ];

    return orderPointsClockwise(
      clipBoxToImageBounds(expanded, imageWidth, imageHeight),
    );
  }

  static img.Image perspectiveTransform(
    img.Image src,
    List<double> srcPoints,
    List<double> dstPoints,
    int dstWidth,
    int dstHeight,
  ) {
    final matrix = computePerspectiveTransform(dstPoints, srcPoints);
    final result = img.Image(width: dstWidth, height: dstHeight);

    for (int y = 0; y < dstHeight; y++) {
      for (int x = 0; x < dstWidth; x++) {
        final srcCoord = applyPerspectiveTransform(
          matrix,
          x.toDouble(),
          y.toDouble(),
        );
        final srcX = srcCoord[0].clamp(0.0, src.width - 1.0);
        final srcY = srcCoord[1].clamp(0.0, src.height - 1.0);

        final x0 = srcX.floor();
        final y0 = srcY.floor();
        final x1 = math.min(x0 + 1, src.width - 1);
        final y1 = math.min(y0 + 1, src.height - 1);

        final fx = srcX - x0;
        final fy = srcY - y0;

        final p00 = src.getPixel(x0, y0);
        final p01 = src.getPixel(x0, y1);
        final p10 = src.getPixel(x1, y0);
        final p11 = src.getPixel(x1, y1);

        final r =
            (p00.r * (1 - fx) * (1 - fy) +
                    p10.r * fx * (1 - fy) +
                    p01.r * (1 - fx) * fy +
                    p11.r * fx * fy)
                .round();
        final g =
            (p00.g * (1 - fx) * (1 - fy) +
                    p10.g * fx * (1 - fy) +
                    p01.g * (1 - fx) * fy +
                    p11.g * fx * fy)
                .round();
        final b =
            (p00.b * (1 - fx) * (1 - fy) +
                    p10.b * fx * (1 - fy) +
                    p01.b * (1 - fx) * fy +
                    p11.b * fx * fy)
                .round();
        final a =
            (p00.a * (1 - fx) * (1 - fy) +
                    p10.a * fx * (1 - fy) +
                    p01.a * (1 - fx) * fy +
                    p11.a * fx * fy)
                .round();

        result.setPixelRgba(x, y, r, g, b, a);
      }
    }

    return result;
  }

  static List<double> computePerspectiveTransform(
    List<double> srcPoints,
    List<double> dstPoints,
  ) {
    final a = <List<double>>[];
    final b = <double>[];

    for (int i = 0; i < 4; i++) {
      final sx = srcPoints[i * 2];
      final sy = srcPoints[i * 2 + 1];
      final dx = dstPoints[i * 2];
      final dy = dstPoints[i * 2 + 1];

      a.add([sx, sy, 1, 0, 0, 0, -dx * sx, -dx * sy]);
      b.add(dx);
      a.add([0, 0, 0, sx, sy, 1, -dy * sx, -dy * sy]);
      b.add(dy);
    }

    return solveLinearSystem(a, b);
  }

  static List<double> solveLinearSystem(List<List<double>> a, List<double> b) {
    final n = a.length;
    final augmented = <List<double>>[];
    for (int i = 0; i < n; i++) {
      augmented.add([...a[i], b[i]]);
    }

    for (int i = 0; i < n; i++) {
      var maxRow = i;
      for (int k = i + 1; k < n; k++) {
        if (augmented[k][i].abs() > augmented[maxRow][i].abs()) {
          maxRow = k;
        }
      }
      final temp = augmented[i];
      augmented[i] = augmented[maxRow];
      augmented[maxRow] = temp;

      for (int k = i + 1; k < n; k++) {
        final factor = augmented[k][i] / augmented[i][i];
        for (int j = i; j <= n; j++) {
          augmented[k][j] -= factor * augmented[i][j];
        }
      }
    }

    final x = List<double>.filled(n, 0);
    for (int i = n - 1; i >= 0; i--) {
      x[i] = augmented[i][n];
      for (int j = i + 1; j < n; j++) {
        x[i] -= augmented[i][j] * x[j];
      }
      x[i] /= augmented[i][i];
    }

    return x;
  }

  static List<double> applyPerspectiveTransform(
    List<double> matrix,
    double x,
    double y,
  ) {
    final w = matrix[6] * x + matrix[7] * y + 1;
    final px = (matrix[0] * x + matrix[1] * y + matrix[2]) / w;
    final py = (matrix[3] * x + matrix[4] * y + matrix[5]) / w;
    return [px, py];
  }

  static double distance(Point p1, Point p2) {
    final dx = p2.x - p1.x;
    final dy = p2.y - p1.y;
    return math.sqrt(dx * dx + dy * dy);
  }

  static List<Point> clipBoxToImageBounds(
    List<Point> points,
    int imageWidth,
    int imageHeight,
  ) {
    return points.map((point) {
      return Point(
        point.x.clamp(0.0, imageWidth - 1.0),
        point.y.clamp(0.0, imageHeight - 1.0),
      );
    }).toList();
  }

  static Point? normalize(Point vector) {
    final length = math.sqrt(vector.x * vector.x + vector.y * vector.y);
    if (length == 0) {
      return null;
    }
    return Point(vector.x / length, vector.y / length);
  }

  static Float32List imageToTensor(
    img.Image image, {
    required int targetHeight,
    required int targetWidth,
    required List<double> mean,
    required List<double> std,
    required double scale,
    required bool bgrOrder,
  }) {
    final resized = img.copyResize(
      image,
      width: targetWidth,
      height: targetHeight,
      interpolation: img.Interpolation.linear,
    );

    final tensor = Float32List(3 * targetHeight * targetWidth);
    final channelStride = targetHeight * targetWidth;

    for (int y = 0; y < targetHeight; y++) {
      for (int x = 0; x < targetWidth; x++) {
        final pixel = resized.getPixel(x, y);
        final r = pixel.r.toDouble() * scale;
        final g = pixel.g.toDouble() * scale;
        final b = pixel.b.toDouble() * scale;

        final pixelIndex = y * targetWidth + x;

        if (bgrOrder) {
          tensor[pixelIndex] = (b - mean[0]) / std[0];
          tensor[pixelIndex + channelStride] = (g - mean[1]) / std[1];
          tensor[pixelIndex + 2 * channelStride] = (r - mean[2]) / std[2];
        } else {
          tensor[pixelIndex] = (r - mean[0]) / std[0];
          tensor[pixelIndex + channelStride] = (g - mean[1]) / std[1];
          tensor[pixelIndex + 2 * channelStride] = (b - mean[2]) / std[2];
        }
      }
    }

    return tensor;
  }
}
