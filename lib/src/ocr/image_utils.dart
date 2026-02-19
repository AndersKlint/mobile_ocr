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

    // Use axis-aligned bounding box instead of perspective transform
    // This is more robust and works well for most text
    double minX = points[0].x;
    double maxX = points[0].x;
    double minY = points[0].y;
    double maxY = points[0].y;
    for (final p in points) {
      if (p.x < minX) minX = p.x;
      if (p.x > maxX) maxX = p.x;
      if (p.y < minY) minY = p.y;
      if (p.y > maxY) maxY = p.y;
    }

    final width = (maxX - minX).toInt().clamp(1, 10000);
    final height = (maxY - minY).toInt().clamp(1, 10000);

    final cropped = img.copyCrop(
      bitmap,
      x: minX.toInt().clamp(0, bitmap.width - 1),
      y: minY.toInt().clamp(0, bitmap.height - 1),
      width: width.clamp(
        1,
        bitmap.width - minX.toInt().clamp(0, bitmap.width - 1),
      ),
      height: height.clamp(
        1,
        bitmap.height - minY.toInt().clamp(0, bitmap.height - 1),
      ),
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
        final srcX = srcCoord[0];
        final srcY = srcCoord[1];

        // Bilinear interpolation
        final x0 = srcX.floor();
        final y0 = srcY.floor();
        final x1 = x0 + 1;
        final y1 = y0 + 1;

        final fx = srcX - x0;
        final fy = srcY - y0;

        if (x0 >= 0 && x1 < src.width && y0 >= 0 && y1 < src.height) {
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

          result.setPixel(x, y, img.ColorRgb8(r, g, b));
        } else if (x0 >= 0 && x0 < src.width && y0 >= 0 && y0 < src.height) {
          // Nearest neighbor for edge pixels
          result.setPixel(x, y, src.getPixel(x0, y0));
        }
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
