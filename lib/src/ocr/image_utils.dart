import 'dart:math' as math;
import 'dart:typed_data';
import 'package:image/image.dart' as img;
import 'types.dart';

class ImageUtils {
  static List<Point> orderPointsClockwise(List<Point> points) {
    if (points.length != 4) {
      return points;
    }

    final centerX = points.map((p) => p.x).reduce((a, b) => a + b) / 4;
    final centerY = points.map((p) => p.y).reduce((a, b) => a + b) / 4;

    final sortedPoints = List<Point>.from(points)
      ..sort((a, b) {
        final angleA = math.atan2(a.y - centerY, a.x - centerX);
        final angleB = math.atan2(b.y - centerY, b.x - centerX);
        return angleA.compareTo(angleB);
      });

    int topLeftIndex = 0;
    double minSum = sortedPoints[0].x + sortedPoints[0].y;
    for (int i = 1; i < 4; i++) {
      final sum = sortedPoints[i].x + sortedPoints[i].y;
      if (sum < minSum) {
        minSum = sum;
        topLeftIndex = i;
      }
    }

    final orderedPoints = <Point>[];
    for (int i = 0; i < 4; i++) {
      orderedPoints.add(sortedPoints[(topLeftIndex + i) % 4]);
    }

    return orderedPoints;
  }

  static img.Image cropTextRegion(img.Image bitmap, List<Point> points) {
    if (points.length != 4) {
      throw ArgumentError('Expected 4 points for text region');
    }

    final width = math
        .max(distance(points[0], points[1]), distance(points[2], points[3]))
        .toInt()
        .clamp(1, 10000);

    final height = math
        .max(distance(points[0], points[3]), distance(points[1], points[2]))
        .toInt()
        .clamp(1, 10000);

    final srcPoints = [
      points[0].x,
      points[0].y,
      points[1].x,
      points[1].y,
      points[2].x,
      points[2].y,
      points[3].x,
      points[3].y,
    ];

    final dstPoints = [
      0.0,
      0.0,
      width.toDouble(),
      0.0,
      width.toDouble(),
      height.toDouble(),
      0.0,
      height.toDouble(),
    ];

    final cropped = perspectiveTransform(
      bitmap,
      srcPoints,
      dstPoints,
      width,
      height,
    );

    if (height / width >= 1.5) {
      return img.copyRotate(cropped, angle: 90);
    }

    return cropped;
  }

  static img.Image perspectiveTransform(
    img.Image src,
    List<double> srcPoints,
    List<double> dstPoints,
    int dstWidth,
    int dstHeight,
  ) {
    final matrix = computePerspectiveTransform(srcPoints, dstPoints);
    final result = img.Image(width: dstWidth, height: dstHeight);

    for (int y = 0; y < dstHeight; y++) {
      for (int x = 0; x < dstWidth; x++) {
        final srcCoord = applyPerspectiveTransform(
          matrix,
          x.toDouble(),
          y.toDouble(),
        );
        final srcX = srcCoord[0].toInt().clamp(0, src.width - 1);
        final srcY = srcCoord[1].toInt().clamp(0, src.height - 1);
        result.setPixel(x, y, src.getPixel(srcX, srcY));
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
