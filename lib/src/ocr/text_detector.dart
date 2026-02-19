import 'dart:math' as math;
import 'dart:typed_data';
import 'package:flutter/foundation.dart';
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';
import 'package:image/image.dart' as img;
import 'types.dart';
import 'image_utils.dart';

class TextDetector {
  static const int limitSideLen = 960;
  static const double thresh = 0.3;
  static const double boxThresh = 0.6;
  static const double unclipRatio = 1.5;
  static const int minSize = 3;
  static const int maxCandidates = 1000;
  static const double epsilon = 1e-6;

  final OrtSession session;

  TextDetector(this.session);

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

    try {
      final inputs = {'x': inputTensor};
      final outputs = await session.run(inputs);
      final output = outputs.values.first;

      await postprocessDetection(
        output: output,
        originalWidth: originalWidth,
        originalHeight: originalHeight,
        resizedWidth: resizedWidth,
        resizedHeight: resizedHeight,
        handler: handler,
      );
    } finally {
      await inputTensor.dispose();
    }
  }

  Future<(OrtValue, int, int)> preprocessImage(img.Image bitmap) async {
    final originalWidth = bitmap.width;
    final originalHeight = bitmap.height;

    final resizeDims = calculateResizeDimensions(originalWidth, originalHeight);
    final resizedWidth = resizeDims.$1;
    final resizedHeight = resizeDims.$2;

    final resized = img.copyResize(
      bitmap,
      width: resizedWidth,
      height: resizedHeight,
      interpolation: img.Interpolation.linear,
    );

    final inputArray = Float32List(1 * 3 * resizedHeight * resizedWidth);

    final mean = [0.485, 0.456, 0.406];
    final std = [0.229, 0.224, 0.225];
    const scale = 1.0 / 255.0;

    for (int y = 0; y < resizedHeight; y++) {
      for (int x = 0; x < resizedWidth; x++) {
        final pixel = resized.getPixel(x, y);
        final b = pixel.b.toDouble() * scale;
        final g = pixel.g.toDouble() * scale;
        final r = pixel.r.toDouble() * scale;

        final pixelIndex = y * resizedWidth + x;

        inputArray[pixelIndex] = (b - mean[0]) / std[0];
        inputArray[pixelIndex + resizedHeight * resizedWidth] =
            (g - mean[1]) / std[1];
        inputArray[pixelIndex + 2 * resizedHeight * resizedWidth] =
            (r - mean[2]) / std[2];
      }
    }

    final shape = [1, 3, resizedHeight, resizedWidth];
    final inputTensor = await OrtValue.fromList(inputArray, shape);

    return (inputTensor, resizedWidth, resizedHeight);
  }

  (int, int) calculateResizeDimensions(int width, int height) {
    final maxSide = width > height ? width : height;
    final ratio = maxSide > limitSideLen ? limitSideLen / maxSide : 1.0;

    var resizedWidth = (width * ratio).round().clamp(1, 10000);
    var resizedHeight = (height * ratio).round().clamp(1, 10000);

    resizedWidth = (((resizedWidth + 31) / 32).floor() * 32).clamp(32, 10000);
    resizedHeight = (((resizedHeight + 31) / 32).floor() * 32).clamp(32, 10000);

    return (resizedWidth, resizedHeight);
  }

  Future<void> postprocessDetection({
    required OrtValue output,
    required int originalWidth,
    required int originalHeight,
    required int resizedWidth,
    required int resizedHeight,
    required bool Function(TextBox, double) handler,
  }) async {
    final outputData = await output.asFlattenedList();

    final probMap = List.generate(
      resizedHeight,
      (y) => List.generate(resizedWidth, (x) {
        final idx = y * resizedWidth + x;
        return (outputData[idx] as num).toDouble();
      }),
    );

    final binaryMap = List.generate(
      resizedHeight,
      (y) => List.generate(resizedWidth, (x) => probMap[y][x] > thresh),
    );

    final components = extractConnectedComponents(binaryMap)
      ..sort((a, b) => b.length.compareTo(a.length));

    final topComponents = components.take(maxCandidates).toList();

    final scaleX = originalWidth / resizedWidth;
    final scaleY = originalHeight / resizedHeight;

    for (final component in topComponents) {
      if (component.length < 4) continue;

      final hull = convexHull(component);
      if (hull.length < 3) continue;

      final rect = minimumAreaRectangle(hull, pointsAreConvex: true);
      if (rect.isEmpty) continue;

      final score = calculateBoxScore(probMap, rect);
      if (score < boxThresh) continue;

      final unclippedPolygon = unclipBox(rect, unclipRatio);
      if (unclippedPolygon.isEmpty) continue;

      final expandedRect = minimumAreaRectangle(
        unclippedPolygon,
        pointsAreConvex: false,
      );
      if (expandedRect.isEmpty) continue;

      final minSide = getMinSide(expandedRect);
      if (minSide < minSize) continue;

      final clippedRect = ImageUtils.clipBoxToImageBounds(
        expandedRect,
        resizedWidth,
        resizedHeight,
      );

      final scaledPoints = clippedRect.map((point) {
        return Point(point.x * scaleX, point.y * scaleY);
      }).toList();

      final orderedPoints = ImageUtils.orderPointsClockwise(scaledPoints);
      final shouldBreak = handler(TextBox(orderedPoints), score);
      if (shouldBreak) break;
    }
  }

  List<List<Point>> extractConnectedComponents(List<List<bool>> binaryMap) {
    final height = binaryMap.length;
    final width = height > 0 ? binaryMap[0].length : 0;
    final visited = List.generate(height, (_) => List.filled(width, false));
    final components = <List<Point>>[];
    final stack = <(int, int)>[];

    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        if (!binaryMap[y][x] || visited[y][x]) continue;

        final points = <Point>[];
        stack.clear();
        stack.add((x, y));
        visited[y][x] = true;

        while (stack.isNotEmpty) {
          final (cx, cy) = stack.removeLast();
          points.add(Point(cx.toDouble(), cy.toDouble()));

          for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
              if (dx == 0 && dy == 0) continue;
              final nx = cx + dx;
              final ny = cy + dy;
              if (nx >= 0 &&
                  nx < width &&
                  ny >= 0 &&
                  ny < height &&
                  binaryMap[ny][nx] &&
                  !visited[ny][nx]) {
                visited[ny][nx] = true;
                stack.add((nx, ny));
              }
            }
          }
        }

        components.add(points);
      }
    }

    return components;
  }

  double calculateBoxScore(List<List<double>> probMap, List<Point> polygon) {
    if (polygon.isEmpty) return 0;

    var minX = polygon.map((p) => p.x).reduce(math.min).floor();
    var maxX = polygon.map((p) => p.x).reduce(math.max).ceil();
    var minY = polygon.map((p) => p.y).reduce(math.min).floor();
    var maxY = polygon.map((p) => p.y).reduce(math.max).ceil();

    minX = minX.clamp(0, probMap[0].length - 1);
    maxX = maxX.clamp(0, probMap[0].length - 1);
    minY = minY.clamp(0, probMap.length - 1);
    maxY = maxY.clamp(0, probMap.length - 1);

    if (maxX < minX || maxY < minY) return 0;

    double sum = 0;
    int count = 0;

    for (int y = minY; y <= maxY; y++) {
      for (int x = minX; x <= maxX; x++) {
        if (isPointInsideQuad(x + 0.5, y + 0.5, polygon)) {
          sum += probMap[y][x];
          count++;
        }
      }
    }

    return count > 0 ? sum / count : 0;
  }

  bool isPointInsideQuad(double x, double y, List<Point> quad) {
    if (quad.length < 3) return false;

    bool hasPositive = false;
    bool hasNegative = false;

    for (int i = 0; i < quad.length; i++) {
      final p1 = quad[i];
      final p2 = quad[(i + 1) % quad.length];
      final cross = (p2.x - p1.x) * (y - p1.y) - (p2.y - p1.y) * (x - p1.x);
      if (cross > 0) {
        hasPositive = true;
      } else if (cross < 0) {
        hasNegative = true;
      }
      if (hasPositive && hasNegative) return false;
    }

    return true;
  }

  List<Point> convexHull(List<Point> points) {
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

  double crossProduct(Point o, Point a, Point b) {
    return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
  }

  List<Point> minimumAreaRectangle(
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

  Point? normalizeVector(Point from, Point to) {
    final dx = to.x - from.x;
    final dy = to.y - from.y;
    final length = math.sqrt(dx * dx + dy * dy);
    if (length < epsilon) return null;
    return Point(dx / length, dy / length);
  }

  List<Point> axisAlignedBoundingBox(List<Point> points) {
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

  List<Point> unclipBox(List<Point> box, double unclipRatio) {
    if (box.length < 3) return [];

    final area = polygonSignedArea(box);
    final perimeter = polygonPerimeter(box);
    if (perimeter <= epsilon) return [];

    final offset = area.abs() * unclipRatio / perimeter;
    if (offset <= epsilon) return box;

    final expanded = offsetPolygon(box, offset);
    return expanded.length >= 3 ? expanded : [];
  }

  double getMinSide(List<Point> box) {
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

  double polygonSignedArea(List<Point> points) {
    double area = 0;
    for (int i = 0; i < points.length; i++) {
      final j = (i + 1) % points.length;
      area += points[i].x * points[j].y - points[j].x * points[i].y;
    }
    return area / 2;
  }

  double polygonPerimeter(List<Point> points) {
    double perimeter = 0;
    for (int i = 0; i < points.length; i++) {
      final j = (i + 1) % points.length;
      perimeter += ImageUtils.distance(points[i], points[j]);
    }
    return perimeter;
  }

  List<Point> offsetPolygon(List<Point> points, double offset) {
    final count = points.length;
    if (count < 3) return [];

    final isCounterClockwise = polygonSignedArea(points) > 0;
    final result = <Point>[];

    for (int i = 0; i < count; i++) {
      final prev = points[(i - 1 + count) % count];
      final curr = points[i];
      final next = points[(i + 1) % count];

      final edge1 = Point(curr.x - prev.x, curr.y - prev.y);
      final edge2 = Point(next.x - curr.x, next.y - curr.y);

      final dir1 = normalize(edge1);
      final dir2 = normalize(edge2);
      if (dir1 == null || dir2 == null) continue;

      final normal1 = isCounterClockwise
          ? Point(dir1.y, -dir1.x)
          : Point(-dir1.y, dir1.x);
      final normal2 = isCounterClockwise
          ? Point(dir2.y, -dir2.x)
          : Point(-dir2.y, dir2.x);

      final offsetPoint1 = Point(
        curr.x + normal1.x * offset,
        curr.y + normal1.y * offset,
      );
      final offsetPoint2 = Point(
        curr.x + normal2.x * offset,
        curr.y + normal2.y * offset,
      );

      final intersection = intersectLines(
        offsetPoint1,
        dir1,
        offsetPoint2,
        dir2,
      );
      result.add(intersection ?? Point(curr.x, curr.y));
    }

    return result;
  }

  Point? normalize(Point vector) {
    final length = math.sqrt(vector.x * vector.x + vector.y * vector.y);
    if (length < epsilon) return null;
    return Point(vector.x / length, vector.y / length);
  }

  Point? intersectLines(
    Point point,
    Point direction,
    Point otherPoint,
    Point otherDirection,
  ) {
    final cross =
        direction.x * otherDirection.y - direction.y * otherDirection.x;
    if (cross.abs() < epsilon) {
      return null;
    }

    final diffX = otherPoint.x - point.x;
    final diffY = otherPoint.y - point.y;
    final t = (diffX * otherDirection.y - diffY * otherDirection.x) / cross;

    return Point(point.x + direction.x * t, point.y + direction.y * t);
  }
}
