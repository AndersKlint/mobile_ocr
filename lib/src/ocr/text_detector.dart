import 'dart:math' as math;
import 'dart:typed_data';
import 'package:clipper2/clipper2.dart' as clipper;
import 'package:image/image.dart' as img;
import 'package:onnxruntime_v2/onnxruntime_v2.dart';
import 'types.dart';
import 'image_utils.dart';
import 'fast_image_loader.dart';
import 'fast_tensor_reader.dart';
import 'ocr_debug_dumper.dart';

class TextDetector {
  static const int limitSideLen = 1920;
  static const String limitType = 'max';
  static const double thresh = 0.5;
  static const double boxThresh = 0.6;
  static const double unclipRatio = 1.3;
  static const int minSize = 3;
  static const int maxCandidates = 1000;
  static const double epsilon = 1e-6;

  final OrtSession session;
  final OcrDebugSession? debugSession;

  TextDetector(this.session, {this.debugSession});

  Future<List<TextBox>> detect(img.Image bitmap) async {
    final boxes = <TextBox>[];
    await runDetection(bitmap, (box, _) {
      boxes.add(box);
      return false;
    });

    if (boxes.isEmpty) {
      return [];
    }

    return sortBoxes(boxes);
  }

  Future<DetectionStageSummary> collectHighConfidenceDetections(
    img.Image bitmap, {
    required double minimumDetectionConfidence,
    required int maxCandidates,
  }) async {
    int examined = 0;
    double maxScore = double.negativeInfinity;
    final candidates = <DetectionCandidate>[];

    await runDetection(bitmap, (box, score) {
      examined++;
      if (score > maxScore) {
        maxScore = score;
      }
      final meetsThreshold = score >= minimumDetectionConfidence;
      if (meetsThreshold) {
        candidates.add(DetectionCandidate(box, score));
        return candidates.length >= maxCandidates;
      }
      return false;
    });

    final bestScore = examined == 0 ? null : maxScore;
    return DetectionStageSummary(
      examinedDetections: examined,
      maxDetectionScore: bestScore,
      candidates: candidates,
    );
  }

  Future<void> runDetection(
    img.Image bitmap,
    bool Function(TextBox, double) handler,
  ) async {
    final originalWidth = bitmap.width;
    final originalHeight = bitmap.height;

    final preprocessResult = await preprocessImage(bitmap);
    final inputTensor = preprocessResult.$1;
    final resizedWidth = preprocessResult.$2;
    final resizedHeight = preprocessResult.$3;
    final resizedImage = preprocessResult.$4;

    if (debugSession != null && resizedImage != null) {
      await debugSession!.saveImage('01_detector/input.png', resizedImage);
      await debugSession!.writeJson('01_detector/meta.json', {
        'originalWidth': originalWidth,
        'originalHeight': originalHeight,
        'resizedWidth': resizedWidth,
        'resizedHeight': resizedHeight,
        'limitSideLen': limitSideLen,
        'limitType': limitType,
        'thresh': thresh,
        'boxThresh': boxThresh,
        'unclipRatio': unclipRatio,
      });
    }

    try {
      final inputs = {'x': inputTensor};
      final runOptions = OrtRunOptions();
      final outputs = session.run(runOptions, inputs);
      runOptions.release();
      final output = outputs[0];

      if (output != null) {
        await postprocessDetection(
          output: output,
          originalWidth: originalWidth,
          originalHeight: originalHeight,
          resizedWidth: resizedWidth,
          resizedHeight: resizedHeight,
          handler: handler,
        );
        output.release();
      }
    } finally {
      inputTensor.release();
    }
  }

  Future<(OrtValueTensor, int, int, img.Image?)> preprocessImage(
    img.Image bitmap,
  ) async {
    final originalWidth = bitmap.width;
    final originalHeight = bitmap.height;

    final resizeDims = calculateResizeDimensions(originalWidth, originalHeight);
    final resizedWidth = resizeDims.$1;
    final resizedHeight = resizeDims.$2;

    final mean = [0.485, 0.456, 0.406];
    final std = [0.229, 0.224, 0.225];

    final inputArray = await FastImageLoader.imageToTensor(
      bitmap,
      targetWidth: resizedWidth,
      targetHeight: resizedHeight,
      mean: mean,
      std: std,
      bgrOrder: true,
    );

    if (inputArray == null) {
      throw StateError('Failed to preprocess image');
    }

    final resizedImage = debugSession == null
        ? null
        : img.copyResize(
            bitmap,
            width: resizedWidth,
            height: resizedHeight,
            interpolation: img.Interpolation.linear,
          );

    final shape = [1, 3, resizedHeight, resizedWidth];
    final inputTensor = OrtValueTensor.createTensorWithDataList(
      inputArray,
      shape,
    );

    return (inputTensor, resizedWidth, resizedHeight, resizedImage);
  }

  static (int, int) calculateResizeDimensions(int width, int height) {
    final maxSide = width > height ? width : height;
    final minSide = width < height ? width : height;

    double ratio;
    switch (limitType) {
      case 'max':
        ratio = maxSide > limitSideLen ? limitSideLen / maxSide : 1.0;
        break;
      case 'min':
        ratio = minSide < limitSideLen ? limitSideLen / minSide : 1.0;
        break;
      case 'resize_long':
        ratio = limitSideLen / maxSide;
        break;
      default:
        throw StateError('Unsupported limitType: $limitType');
    }

    var resizedWidth = (width * ratio).toInt().clamp(1, 10000);
    var resizedHeight = (height * ratio).toInt().clamp(1, 10000);

    resizedWidth = math.max(_roundToNearestMultipleOf32(resizedWidth), 32);
    resizedHeight = math.max(_roundToNearestMultipleOf32(resizedHeight), 32);

    return (resizedWidth, resizedHeight);
  }

  static int _roundToNearestMultipleOf32(int value) {
    final scaled = value / 32.0;
    final lower = scaled.floor();
    final fraction = scaled - lower;

    if (fraction < 0.5) {
      return lower * 32;
    }
    if (fraction > 0.5) {
      return (lower + 1) * 32;
    }
    return (lower.isEven ? lower : lower + 1) * 32;
  }

  Future<void> postprocessDetection({
    required OrtValue output,
    required int originalWidth,
    required int originalHeight,
    required int resizedWidth,
    required int resizedHeight,
    required bool Function(TextBox, double) handler,
  }) async {
    final prob = FastTensorReader.asFloat32List(output);
    if (prob == null || prob.isEmpty) return;

    await _dumpProbabilityMap(prob, resizedWidth, resizedHeight);

    final binaryMap = buildBinaryMap(prob, resizedWidth, resizedHeight);
    final contours = traceContours(binaryMap);
    contours.sort((a, b) => polygonArea(b).compareTo(polygonArea(a)));
    await debugSession?.writeJson(
      '01_detector/contours.json',
      contours
          .map(
            (contour) => contour
                .map((point) => {'x': point.x, 'y': point.y})
                .toList(growable: false),
          )
          .toList(growable: false),
    );

    final topContours = contours.take(maxCandidates).toList();

    final scaleX = originalWidth / resizedWidth;
    final scaleY = originalHeight / resizedHeight;

    final acceptedBoxes = <Map<String, Object?>>[];
    for (final contour in topContours) {
      if (contour.length < 4) continue;

      final miniBoxResult = getMiniBox(contour);
      final rect = miniBoxResult.$1;
      final shortSide = miniBoxResult.$2;
      if (rect.isEmpty || shortSide < minSize) continue;

      final score = calculateBoxScore(prob, resizedWidth, resizedHeight, rect);
      if (score < boxThresh) continue;

      final unclippedPolygon = unclipBox(rect, unclipRatio);
      if (unclippedPolygon.isEmpty) continue;

      final expandedMiniBox = getMiniBox(unclippedPolygon);
      final expandedRect = expandedMiniBox.$1;
      final expandedShortSide = expandedMiniBox.$2;
      if (expandedRect.isEmpty || expandedShortSide < minSize + 2) continue;

      final clippedRect = ImageUtils.clipBoxToImageBounds(
        expandedRect,
        resizedWidth,
        resizedHeight,
      );

      final scaledPoints = clippedRect.map((point) {
        return Point(point.x * scaleX, point.y * scaleY);
      }).toList();

      final orderedPoints = ImageUtils.orderPointsClockwise(scaledPoints);
      acceptedBoxes.add({
        'score': score,
        'points': orderedPoints
            .map((point) => {'x': point.x, 'y': point.y})
            .toList(growable: false),
      });
      final shouldBreak = handler(TextBox(orderedPoints), score);
      if (shouldBreak) break;
    }

    await debugSession?.writeJson('01_detector/boxes.json', acceptedBoxes);
  }

  Future<void> _dumpProbabilityMap(
    Float32List prob,
    int resizedWidth,
    int resizedHeight,
  ) async {
    if (debugSession == null) {
      return;
    }

    final probabilityImage = img.Image(
      width: resizedWidth,
      height: resizedHeight,
    );
    final binaryImage = img.Image(width: resizedWidth, height: resizedHeight);

    for (int y = 0; y < resizedHeight; y++) {
      for (int x = 0; x < resizedWidth; x++) {
        final value = prob[y * resizedWidth + x].clamp(0.0, 1.0);
        final grayscale = (value * 255).round();
        probabilityImage.setPixelRgba(
          x,
          y,
          grayscale,
          grayscale,
          grayscale,
          255,
        );
        final binaryValue = value > thresh ? 255 : 0;
        binaryImage.setPixelRgba(
          x,
          y,
          binaryValue,
          binaryValue,
          binaryValue,
          255,
        );
      }
    }

    await debugSession!.saveImage(
      '01_detector/probability_map.png',
      probabilityImage,
    );
    await debugSession!.saveImage('01_detector/binary_map.png', binaryImage);
  }

  static List<List<bool>> buildBinaryMap(
    Float32List prob,
    int width,
    int height,
  ) {
    return List.generate(
      height,
      (y) => List.generate(width, (x) {
        final idx = y * width + x;
        return idx < prob.length && prob[idx] > thresh;
      }),
    );
  }

  static List<List<Point>> traceContours(List<List<bool>> binaryMap) {
    final height = binaryMap.length;
    final width = height > 0 ? binaryMap[0].length : 0;
    final visited = List.generate(height, (_) => List.filled(width, false));
    final contours = <List<Point>>[];

    bool isBoundaryPixel(int x, int y) {
      if (!binaryMap[y][x]) return false;
      for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
          if (dx == 0 && dy == 0) continue;
          final nx = x + dx;
          final ny = y + dy;
          if (nx < 0 || nx >= width || ny < 0 || ny >= height) {
            return true;
          }
          if (!binaryMap[ny][nx]) {
            return true;
          }
        }
      }
      return false;
    }

    final stack = <(int, int)>[];
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        if (!binaryMap[y][x] || visited[y][x]) continue;

        final contour = <Point>[];
        stack.clear();
        stack.add((x, y));
        visited[y][x] = true;

        while (stack.isNotEmpty) {
          final (cx, cy) = stack.removeLast();
          if (isBoundaryPixel(cx, cy)) {
            contour.add(Point(cx.toDouble(), cy.toDouble()));
          }

          for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
              if (dx == 0 && dy == 0) continue;
              final nx = cx + dx;
              final ny = cy + dy;
              if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
              if (!binaryMap[ny][nx] || visited[ny][nx]) continue;
              visited[ny][nx] = true;
              stack.add((nx, ny));
            }
          }
        }

        if (contour.length >= 4) {
          contours.add(convexHull(contour));
        }
      }
    }

    return contours;
  }

  static (List<Point>, double) getMiniBox(List<Point> contour) {
    final rect = minimumAreaRectangle(contour, pointsAreConvex: false);
    if (rect.isEmpty) {
      return ([], 0);
    }
    return (ImageUtils.orderPointsClockwise(rect), getMinSide(rect));
  }

  static double calculateBoxScore(
    Float32List prob,
    int width,
    int height,
    List<Point> polygon,
  ) {
    if (polygon.isEmpty) return 0;

    var minX = polygon.map((p) => p.x).reduce(math.min).floor();
    var maxX = polygon.map((p) => p.x).reduce(math.max).ceil();
    var minY = polygon.map((p) => p.y).reduce(math.min).floor();
    var maxY = polygon.map((p) => p.y).reduce(math.max).ceil();

    minX = minX.clamp(0, width - 1);
    maxX = maxX.clamp(0, width - 1);
    minY = minY.clamp(0, height - 1);
    maxY = maxY.clamp(0, height - 1);

    if (maxX < minX || maxY < minY) return 0;

    final maskWidth = maxX - minX + 1;
    final maskHeight = maxY - minY + 1;
    final maskImage = img.Image(width: maskWidth, height: maskHeight);
    img.fill(maskImage, color: img.ColorRgb8(0, 0, 0));
    img.fillPolygon(
      maskImage,
      vertices: polygon
          .map((point) => img.Point(point.x - minX, point.y - minY))
          .toList(growable: false),
      color: img.ColorRgb8(255, 255, 255),
    );

    double sum = 0;
    int count = 0;
    for (int y = 0; y < maskHeight; y++) {
      for (int x = 0; x < maskWidth; x++) {
        if (maskImage.getPixel(x, y).r > 0) {
          sum += prob[(y + minY) * width + (x + minX)];
          count++;
        }
      }
    }

    return count > 0 ? sum / count : 0;
  }

  static List<Point> convexHull(List<Point> points) {
    if (points.length < 3) return points;

    final sorted = List<Point>.from(points)
      ..sort((a, b) {
        final cmp = a.x.compareTo(b.x);
        return cmp != 0 ? cmp : a.y.compareTo(b.y);
      });

    final lower = <Point>[];
    final upper = <Point>[];

    for (final point in sorted) {
      while (lower.length >= 2 &&
          crossProduct(
                lower[lower.length - 2],
                lower[lower.length - 1],
                point,
              ) <=
              0) {
        lower.removeLast();
      }
      lower.add(point);
    }

    for (final point in sorted.reversed) {
      while (upper.length >= 2 &&
          crossProduct(
                upper[upper.length - 2],
                upper[upper.length - 1],
                point,
              ) <=
              0) {
        upper.removeLast();
      }
      upper.add(point);
    }

    lower.removeLast();
    upper.removeLast();
    return [...lower, ...upper];
  }

  static double crossProduct(Point o, Point a, Point b) {
    return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
  }

  static List<Point> minimumAreaRectangle(
    List<Point> points, {
    bool pointsAreConvex = false,
  }) {
    final hull = pointsAreConvex ? points : convexHull(points);
    if (hull.length < 3) return [];

    List<Point> bestRect = [];
    var minArea = double.infinity;

    for (int i = 0; i < hull.length; i++) {
      final p1 = hull[i];
      final p2 = hull[(i + 1) % hull.length];
      final edgeVec = normalizeVector(p1, p2);
      if (edgeVec == null) continue;
      final normal = Point(-edgeVec.y, edgeVec.x);

      var minProj = double.infinity;
      var maxProj = double.negativeInfinity;
      var minOrth = double.infinity;
      var maxOrth = double.negativeInfinity;

      for (final pt in hull) {
        final relX = pt.x - p1.x;
        final relY = pt.y - p1.y;
        final projection = relX * edgeVec.x + relY * edgeVec.y;
        final orthProjection = relX * normal.x + relY * normal.y;

        if (projection < minProj) minProj = projection;
        if (projection > maxProj) maxProj = projection;
        if (orthProjection < minOrth) minOrth = orthProjection;
        if (orthProjection > maxOrth) maxOrth = orthProjection;
      }

      final width = maxProj - minProj;
      final height = maxOrth - minOrth;
      final area = width * height;

      if (area < minArea && width > 1e-3 && height > 1e-3) {
        minArea = area;

        final corner0 = Point(
          p1.x + edgeVec.x * minProj + normal.x * minOrth,
          p1.y + edgeVec.y * minProj + normal.y * minOrth,
        );
        final corner1 = Point(
          p1.x + edgeVec.x * maxProj + normal.x * minOrth,
          p1.y + edgeVec.y * maxProj + normal.y * minOrth,
        );
        final corner2 = Point(
          p1.x + edgeVec.x * maxProj + normal.x * maxOrth,
          p1.y + edgeVec.y * maxProj + normal.y * maxOrth,
        );
        final corner3 = Point(
          p1.x + edgeVec.x * minProj + normal.x * maxOrth,
          p1.y + edgeVec.y * minProj + normal.y * maxOrth,
        );

        bestRect = [corner0, corner1, corner2, corner3];
      }
    }

    return bestRect.isEmpty ? axisAlignedBoundingBox(hull) : bestRect;
  }

  static Point? normalizeVector(Point from, Point to) {
    final dx = to.x - from.x;
    final dy = to.y - from.y;
    final length = math.sqrt(dx * dx + dy * dy);
    if (length < epsilon) return null;
    return Point(dx / length, dy / length);
  }

  static List<Point> axisAlignedBoundingBox(List<Point> points) {
    if (points.isEmpty) return [];

    final minX = points.map((p) => p.x).reduce(math.min);
    final maxX = points.map((p) => p.x).reduce(math.max);
    final minY = points.map((p) => p.y).reduce(math.min);
    final maxY = points.map((p) => p.y).reduce(math.max);

    return [
      Point(minX, minY),
      Point(maxX, minY),
      Point(maxX, maxY),
      Point(minX, maxY),
    ];
  }

  List<TextBox> sortBoxes(List<TextBox> boxes) {
    if (boxes.isEmpty) return [];

    final sortedByTop = List<TextBox>.from(boxes)
      ..sort((a, b) {
        final minYa = a.points.map((p) => p.y).reduce(math.min);
        final minYb = b.points.map((p) => p.y).reduce(math.min);
        return minYa.compareTo(minYb);
      });

    final ordered = <TextBox>[];
    int index = 0;
    while (index < sortedByTop.length) {
      final current = sortedByTop[index];
      final referenceY = current.points.map((p) => p.y).reduce(math.min);
      final group = <TextBox>[];

      int j = index;
      while (j < sortedByTop.length) {
        final candidate = sortedByTop[j];
        final candidateY = candidate.points.map((p) => p.y).reduce(math.min);
        if ((candidateY - referenceY).abs() <= 10) {
          group.add(candidate);
          j++;
        } else {
          break;
        }
      }

      group.sort((a, b) {
        final minxa = a.points.map((p) => p.x).reduce(math.min);
        final minxb = b.points.map((p) => p.x).reduce(math.min);
        return minxa.compareTo(minxb);
      });
      ordered.addAll(group);
      index = j;
    }

    return ordered;
  }

  static List<Point> unclipBox(List<Point> box, double unclipRatio) {
    if (box.length < 3) return [];

    final area = polygonArea(box);
    final perimeter = polygonPerimeter(box);
    if (perimeter <= epsilon || area <= epsilon) return [];

    final distance = area * unclipRatio / perimeter;
    final scale = 100.0;
    final path = box
        .map(
          (point) => clipper.Point64.fromDouble(point.x, point.y, scale: scale),
        )
        .toList(growable: false);

    final offset = clipper.ClipperOffset();
    offset.addPath(
      path,
      joinType: clipper.JoinType.round,
      endType: clipper.EndType.polygon,
    );

    final solution = offset.execute(delta: distance * scale);
    if (solution.isEmpty) {
      return [];
    }

    solution.sort((a, b) => b.area.abs().compareTo(a.area.abs()));
    return solution.first
        .map((point) => Point(point.x / scale, point.y / scale))
        .toList(growable: false);
  }

  static double polygonArea(List<Point> points) {
    return polygonSignedArea(points).abs();
  }

  static double getMinSide(List<Point> box) {
    if (box.length < 2) return 0;
    var minSide = double.infinity;
    for (int i = 0; i < box.length; i++) {
      final next = (i + 1) % box.length;
      final length = ImageUtils.distance(box[i], box[next]);
      if (length < minSide) {
        minSide = length;
      }
    }
    return minSide == double.infinity ? 0 : minSide;
  }

  static double polygonSignedArea(List<Point> points) {
    double area = 0;
    for (int i = 0; i < points.length; i++) {
      final j = (i + 1) % points.length;
      area += points[i].x * points[j].y - points[j].x * points[i].y;
    }
    return area / 2;
  }

  static double polygonPerimeter(List<Point> points) {
    double perimeter = 0;
    for (int i = 0; i < points.length; i++) {
      final j = (i + 1) % points.length;
      perimeter += ImageUtils.distance(points[i], points[j]);
    }
    return perimeter;
  }
}
