import 'dart:math' as math;
import 'dart:typed_data';
import 'package:image/image.dart' as img;
import 'types.dart';

class ImageUtils {
  static const double _verticalTrimMinForegroundRatio = 0.01;
  static const int _verticalTrimMinForegroundPixels = 2;
  static const int _verticalTrimPaddingRows = 2;
  static const int _verticalTrimMinBlankRows = 2;
  static const int _verticalTrimMinContrast = 24;
  static const int _verticalTrimMergeGapRows = 3;
  static const double _portraitCropAspectRatioThreshold = 1.5;
  static const double defaultRecognitionContrastBoost = 0.08;
  static const double defaultRecognitionBrightnessBoost = 0.02;

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

  static img.Image cropTextRegion(
    img.Image bitmap,
    List<Point> points, {
    bool trimWhitespace = false,
    bool preferLandscape = false,
  }) {
    if (points.length != 4) {
      throw ArgumentError('Expected 4 points for text region');
    }

    final ordered = orderPointsClockwise(
      clipBoxToImageBounds(points, bitmap.width, bitmap.height),
    );
    final cropWidth = math
        .max(distance(ordered[0], ordered[1]), distance(ordered[2], ordered[3]))
        .ceil()
        .clamp(1, 10000);
    final cropHeight = math
        .max(distance(ordered[0], ordered[3]), distance(ordered[1], ordered[2]))
        .ceil()
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

    final shouldRotateToLandscape = shouldRotateRecognitionRegion(
      cropWidth: cropWidth.toDouble(),
      cropHeight: cropHeight.toDouble(),
      preferLandscape: preferLandscape,
    );
    final normalized = shouldRotateToLandscape
        ? img.copyRotate(cropped, angle: 90)
        : cropped;

    return trimWhitespace ? trimVerticalWhitespace(normalized) : normalized;
  }

  static img.Image trimVerticalWhitespace(img.Image image) {
    if (image.height <= 4 || image.width <= 0) {
      return image;
    }

    final totalPixels = image.width * image.height;
    final luminanceValues = Uint8List(totalPixels);
    final histogram = List<int>.filled(256, 0);

    var minLuminance = 255;
    var maxLuminance = 0;
    var offset = 0;

    for (int y = 0; y < image.height; y++) {
      for (int x = 0; x < image.width; x++) {
        final pixel = image.getPixel(x, y);
        final luminance = _pixelLuminance(pixel.r, pixel.g, pixel.b);
        luminanceValues[offset++] = luminance;
        histogram[luminance]++;

        if (luminance < minLuminance) minLuminance = luminance;
        if (luminance > maxLuminance) maxLuminance = luminance;
      }
    }

    if (maxLuminance - minLuminance < _verticalTrimMinContrast) {
      return image;
    }

    final threshold = _computeOtsuThreshold(histogram, totalPixels);
    final darkPixels = histogram
        .sublist(0, threshold + 1)
        .fold<int>(0, (sum, count) => sum + count);
    final lightPixels = totalPixels - darkPixels;
    final foregroundIsDark = darkPixels <= lightPixels;

    final minForegroundPixels = math.max(
      _verticalTrimMinForegroundPixels,
      (image.width * _verticalTrimMinForegroundRatio).ceil(),
    );
    final rowForegroundCounts = List<int>.filled(image.height, 0);

    offset = 0;
    for (int y = 0; y < image.height; y++) {
      var count = 0;
      for (int x = 0; x < image.width; x++) {
        final luminance = luminanceValues[offset++];
        final isForeground = foregroundIsDark
            ? luminance <= threshold
            : luminance > threshold;
        if (isForeground) {
          count++;
        }
      }
      rowForegroundCounts[y] = count;
    }

    final activeRows = List<bool>.filled(image.height, false);
    for (int y = 0; y < image.height; y++) {
      var localMax = rowForegroundCounts[y];
      if (y > 0 && rowForegroundCounts[y - 1] > localMax) {
        localMax = rowForegroundCounts[y - 1];
      }
      if (y + 1 < image.height && rowForegroundCounts[y + 1] > localMax) {
        localMax = rowForegroundCounts[y + 1];
      }

      if (localMax >= minForegroundPixels) {
        activeRows[y] = true;
      }
    }

    final bands = _buildVerticalBands(activeRows, rowForegroundCounts);
    if (bands.isEmpty) {
      return image;
    }

    final primaryBand = bands.reduce((best, current) {
      if (current.inkSum != best.inkSum) {
        return current.inkSum > best.inkSum ? current : best;
      }
      if (current.activeRows != best.activeRows) {
        return current.activeRows > best.activeRows ? current : best;
      }
      return current.height > best.height ? current : best;
    });

    final top = math.max(0, primaryBand.start - _verticalTrimPaddingRows);
    final bottom = math.min(
      image.height - 1,
      primaryBand.end + _verticalTrimPaddingRows,
    );
    final trimmedTopRows = top;
    final trimmedBottomRows = image.height - 1 - bottom;

    if (trimmedTopRows < _verticalTrimMinBlankRows &&
        trimmedBottomRows < _verticalTrimMinBlankRows) {
      return image;
    }

    final trimmedHeight = bottom - top + 1;
    if (trimmedHeight <= 0 || trimmedHeight >= image.height) {
      return image;
    }

    return img.copyCrop(
      image,
      x: 0,
      y: top,
      width: image.width,
      height: trimmedHeight,
    );
  }

  static img.Image enhanceRecognitionCrop(
    img.Image image, {
    double contrastBoost = defaultRecognitionContrastBoost,
    double brightnessBoost = defaultRecognitionBrightnessBoost,
  }) {
    if (contrastBoost <= 0 && brightnessBoost <= 0) {
      return image;
    }

    final sourceBytes = _tryGetUint8Bytes(image);
    final sourceChannels = image.numChannels;
    if (sourceBytes != null && (sourceChannels == 3 || sourceChannels == 4)) {
      final adjustedBytes = Uint8List(
        image.width * image.height * sourceChannels,
      );
      final lut = _buildToneAdjustmentLut(
        1.0 + contrastBoost,
        255.0 * brightnessBoost,
      );

      if (sourceChannels == 4) {
        for (int i = 0; i < sourceBytes.length; i += 4) {
          adjustedBytes[i] = lut[sourceBytes[i]];
          adjustedBytes[i + 1] = lut[sourceBytes[i + 1]];
          adjustedBytes[i + 2] = lut[sourceBytes[i + 2]];
          adjustedBytes[i + 3] = sourceBytes[i + 3];
        }
      } else {
        for (int i = 0; i < sourceBytes.length; i += 3) {
          adjustedBytes[i] = lut[sourceBytes[i]];
          adjustedBytes[i + 1] = lut[sourceBytes[i + 1]];
          adjustedBytes[i + 2] = lut[sourceBytes[i + 2]];
        }
      }

      return _imageFromUint8Bytes(
        adjustedBytes,
        width: image.width,
        height: image.height,
        numChannels: sourceChannels,
      );
    }

    final adjusted = img.Image(width: image.width, height: image.height);
    final contrast = 1.0 + contrastBoost;
    final brightnessOffset = 255.0 * brightnessBoost;

    for (int y = 0; y < image.height; y++) {
      for (int x = 0; x < image.width; x++) {
        final pixel = image.getPixel(x, y);
        adjusted.setPixelRgba(
          x,
          y,
          _applyRecognitionToneAdjustment(pixel.r, contrast, brightnessOffset),
          _applyRecognitionToneAdjustment(pixel.g, contrast, brightnessOffset),
          _applyRecognitionToneAdjustment(pixel.b, contrast, brightnessOffset),
          pixel.a,
        );
      }
    }

    return adjusted;
  }

  static int _pixelLuminance(num r, num g, num b) {
    return ((299 * r) + (587 * g) + (114 * b)).round() ~/ 1000;
  }

  static int _applyRecognitionToneAdjustment(
    num value,
    double contrast,
    double brightnessOffset,
  ) {
    final adjusted = ((value - 128.0) * contrast) + 128.0 + brightnessOffset;
    return adjusted.round().clamp(0, 255);
  }

  static List<_VerticalBand> _buildVerticalBands(
    List<bool> activeRows,
    List<int> rowForegroundCounts,
  ) {
    final bands = <_VerticalBand>[];
    _VerticalBand? current;

    for (int y = 0; y < activeRows.length; y++) {
      if (!activeRows[y]) {
        continue;
      }

      if (current == null || y - current.end - 1 > _verticalTrimMergeGapRows) {
        current = _VerticalBand(
          start: y,
          end: y,
          inkSum: rowForegroundCounts[y],
          activeRows: 1,
        );
        bands.add(current);
      } else {
        current.end = y;
        current.inkSum += rowForegroundCounts[y];
        current.activeRows++;
      }
    }

    return bands;
  }

  static int _computeOtsuThreshold(List<int> histogram, int totalPixels) {
    var sum = 0.0;
    for (int i = 0; i < histogram.length; i++) {
      sum += i * histogram[i];
    }

    var sumBackground = 0.0;
    var weightBackground = 0;
    var maxVariance = -1.0;
    var threshold = 0;

    for (int i = 0; i < histogram.length; i++) {
      weightBackground += histogram[i];
      if (weightBackground == 0) {
        continue;
      }

      final weightForeground = totalPixels - weightBackground;
      if (weightForeground == 0) {
        break;
      }

      sumBackground += i * histogram[i];
      final meanBackground = sumBackground / weightBackground;
      final meanForeground = (sum - sumBackground) / weightForeground;
      final variance =
          weightBackground *
          weightForeground *
          math.pow(meanBackground - meanForeground, 2);

      if (variance > maxVariance) {
        maxVariance = variance.toDouble();
        threshold = i;
      }
    }

    return threshold;
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

  static bool shouldRotateRecognitionBox(
    List<Point> points, {
    bool preferLandscape = false,
  }) {
    return shouldRotateRecognitionRegion(
      cropWidth: quadWidth(points),
      cropHeight: quadHeight(points),
      preferLandscape: preferLandscape,
    );
  }

  static bool? inferRecognitionRegionLandscapePreference({
    required double cropWidth,
    required double cropHeight,
  }) {
    if (cropWidth <= 0 || cropHeight <= 0) {
      return null;
    }

    if (cropHeight / cropWidth >= _portraitCropAspectRatioThreshold) {
      return false;
    }
    if (cropWidth / cropHeight >= _portraitCropAspectRatioThreshold) {
      return true;
    }

    return null;
  }

  static bool? inferRecognitionBoxLandscapePreference(List<Point> points) {
    return inferRecognitionRegionLandscapePreference(
      cropWidth: quadWidth(points),
      cropHeight: quadHeight(points),
    );
  }

  static bool shouldRotateRecognitionRegion({
    required double cropWidth,
    required double cropHeight,
    bool preferLandscape = false,
  }) {
    final inferredPreference = inferRecognitionRegionLandscapePreference(
      cropWidth: cropWidth,
      cropHeight: cropHeight,
    );
    if (inferredPreference == null) {
      if (cropWidth <= 0 || cropHeight <= 0) {
        return false;
      }

      return preferLandscape && cropHeight > cropWidth;
    }

    if (!inferredPreference) {
      return true;
    }

    return false;
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

    final sourceBytes = _tryGetUint8Bytes(src);
    final sourceChannels = src.numChannels;
    if (sourceBytes != null && (sourceChannels == 3 || sourceChannels == 4)) {
      final dstBytes = Uint8List(dstWidth * dstHeight * sourceChannels);
      final maxX = src.width - 1.0;
      final maxY = src.height - 1.0;
      final srcRowStride = src.rowStride;

      final m0 = matrix[0];
      final m1 = matrix[1];
      final m2 = matrix[2];
      final m3 = matrix[3];
      final m4 = matrix[4];
      final m5 = matrix[5];
      final m6 = matrix[6];
      final m7 = matrix[7];

      var dstOffset = 0;
      for (int y = 0; y < dstHeight; y++) {
        final yAsDouble = y.toDouble();
        var numeratorX = m1 * yAsDouble + m2;
        var numeratorY = m4 * yAsDouble + m5;
        var denominator = m7 * yAsDouble + 1.0;

        for (int x = 0; x < dstWidth; x++) {
          final srcX = (numeratorX / denominator).clamp(0.0, maxX);
          final srcY = (numeratorY / denominator).clamp(0.0, maxY);

          final x0 = srcX.floor();
          final y0 = srcY.floor();
          final x1 = math.min(x0 + 1, src.width - 1);
          final y1 = math.min(y0 + 1, src.height - 1);

          final fx = srcX - x0;
          final fy = srcY - y0;
          final w00 = (1 - fx) * (1 - fy);
          final w10 = fx * (1 - fy);
          final w01 = (1 - fx) * fy;
          final w11 = fx * fy;

          final base00 = y0 * srcRowStride + x0 * sourceChannels;
          final base01 = y1 * srcRowStride + x0 * sourceChannels;
          final base10 = y0 * srcRowStride + x1 * sourceChannels;
          final base11 = y1 * srcRowStride + x1 * sourceChannels;

          dstBytes[dstOffset] =
              (sourceBytes[base00] * w00 +
                      sourceBytes[base10] * w10 +
                      sourceBytes[base01] * w01 +
                      sourceBytes[base11] * w11)
                  .round();
          dstBytes[dstOffset + 1] =
              (sourceBytes[base00 + 1] * w00 +
                      sourceBytes[base10 + 1] * w10 +
                      sourceBytes[base01 + 1] * w01 +
                      sourceBytes[base11 + 1] * w11)
                  .round();
          dstBytes[dstOffset + 2] =
              (sourceBytes[base00 + 2] * w00 +
                      sourceBytes[base10 + 2] * w10 +
                      sourceBytes[base01 + 2] * w01 +
                      sourceBytes[base11 + 2] * w11)
                  .round();

          if (sourceChannels == 4) {
            dstBytes[dstOffset + 3] =
                (sourceBytes[base00 + 3] * w00 +
                        sourceBytes[base10 + 3] * w10 +
                        sourceBytes[base01 + 3] * w01 +
                        sourceBytes[base11 + 3] * w11)
                    .round();
          }

          dstOffset += sourceChannels;
          numeratorX += m0;
          numeratorY += m3;
          denominator += m6;
        }
      }

      return _imageFromUint8Bytes(
        dstBytes,
        width: dstWidth,
        height: dstHeight,
        numChannels: sourceChannels,
      );
    }

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

  static Uint8List? _tryGetUint8Bytes(img.Image image) {
    final imageData = image.data;
    if (imageData is! img.ImageDataUint8 || image.hasPalette) {
      return null;
    }
    final channels = image.numChannels;
    if (channels != 3 && channels != 4) {
      return null;
    }
    return image.toUint8List();
  }

  static img.Image _imageFromUint8Bytes(
    Uint8List bytes, {
    required int width,
    required int height,
    required int numChannels,
  }) {
    return img.Image.fromBytes(
      width: width,
      height: height,
      bytes: bytes.buffer,
      numChannels: numChannels,
      order: numChannels == 4 ? img.ChannelOrder.rgba : img.ChannelOrder.rgb,
    );
  }

  static Uint8List _buildToneAdjustmentLut(
    double contrast,
    double brightnessOffset,
  ) {
    final lut = Uint8List(256);
    for (int value = 0; value < 256; value++) {
      lut[value] = _applyRecognitionToneAdjustment(
        value,
        contrast,
        brightnessOffset,
      );
    }
    return lut;
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

class _VerticalBand {
  _VerticalBand({
    required this.start,
    required this.end,
    required this.inkSum,
    required this.activeRows,
  });

  int start;
  int end;
  int inkSum;
  int activeRows;

  int get height => end - start + 1;
}
