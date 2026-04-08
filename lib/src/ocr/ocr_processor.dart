import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:onnxruntime_v2/onnxruntime_v2.dart';
import 'package:image/image.dart' as img;
import 'types.dart';
import 'image_utils.dart';
import 'ocr_debug_dumper.dart';
import 'text_detector.dart';
import 'text_recognizer.dart';
import 'text_classifier.dart';

class OcrProcessor {
  static const double minRecognitionScore = 0.7;
  static const double fallbackMinRecognitionScore = 0.5;
  static const double angleAspectRatioThreshold = 0.5;
  static const double lowConfidenceThreshold = 0.55;
  static const int quickCheckMaxCandidates = 3;
  static const List<int> _pageOrientationProbeAngles = [0, 90, 180, 270];

  final OrtSession detectionSession;
  final OrtSession recognitionSession;
  final OrtSession? classificationSession;
  final List<String> characterDict;
  final bool useAngleClassification;
  final OcrDebugSession? debugSession;
  final OcrProcessingOptions processingOptions;

  late final TextDetector _detector;
  late final TextRecognizer _recognizer;
  TextClassifier? _classifier;

  OcrProcessor({
    required this.detectionSession,
    required this.recognitionSession,
    required this.classificationSession,
    required this.characterDict,
    this.useAngleClassification = true,
    this.debugSession,
    this.processingOptions = const OcrProcessingOptions(),
  }) {
    _detector = TextDetector(detectionSession, debugSession: debugSession);
    _recognizer = TextRecognizer(
      recognitionSession,
      characterDict,
      debugSession: debugSession,
    );
    if (useAngleClassification && classificationSession != null) {
      _classifier = TextClassifier(
        classificationSession!,
        debugSession: debugSession,
      );
    }
  }

  static Future<OcrProcessor> create({
    required String detectionModelPath,
    required String recognitionModelPath,
    required String? classificationModelPath,
    required String dictionaryPath,
    bool useAngleClassification = true,
    String? debugDumpDir,
    OcrProcessingOptions processingOptions = const OcrProcessingOptions(),
  }) async {
    OrtEnv.instance.init();

    final sessionOptions = OrtSessionOptions();
    sessionOptions.appendDefaultProviders();

    final detectionBytes = await File(detectionModelPath).readAsBytes();
    final detectionSession = OrtSession.fromBuffer(
      detectionBytes,
      sessionOptions,
    );

    final recognitionBytes = await File(recognitionModelPath).readAsBytes();
    final recognitionSession = OrtSession.fromBuffer(
      recognitionBytes,
      sessionOptions,
    );

    OrtSession? classificationSession;
    if (useAngleClassification && classificationModelPath != null) {
      final classificationBytes = await File(
        classificationModelPath,
      ).readAsBytes();
      classificationSession = OrtSession.fromBuffer(
        classificationBytes,
        sessionOptions,
      );
    }

    final dictFile = File(dictionaryPath);
    final dictLines = await dictFile.readAsLines();
    final characterDict = ['blank', ...dictLines, ' '];
    final debugSession = await OcrDebugSession.create(debugDumpDir);

    return OcrProcessor(
      detectionSession: detectionSession,
      recognitionSession: recognitionSession,
      classificationSession: classificationSession,
      characterDict: characterDict,
      useAngleClassification: useAngleClassification,
      debugSession: debugSession,
      processingOptions: processingOptions,
    );
  }

  Future<OcrResult> processImage(
    img.Image bitmap, {
    bool includeAllConfidenceScores = false,
  }) async {
    await debugSession?.saveImage('00_source/source.png', bitmap);

    final selectedOrientation = await _selectBestPageOrientation(bitmap);
    final workingBitmap = selectedOrientation.angle == 0
        ? bitmap
        : img.copyRotate(bitmap, angle: selectedOrientation.angle);
    await debugSession?.writeJson('00_source/page_orientation.json', {
      'selectedAngle': selectedOrientation.angle,
      'candidateCount': selectedOrientation.candidateCount,
      'maxDetectionScore': selectedOrientation.maxDetectionScore,
    });

    final rawResult = await _processOrientedImage(
      workingBitmap,
      includeAllConfidenceScores: includeAllConfidenceScores,
    );

    if (selectedOrientation.angle == 0 || rawResult.boxes.isEmpty) {
      return rawResult;
    }

    return _mapResultToOriginalOrientation(
      rawResult,
      angle: selectedOrientation.angle,
      originalWidth: bitmap.width,
      originalHeight: bitmap.height,
    );
  }

  Future<OcrResult> _processOrientedImage(
    img.Image bitmap, {
    required bool includeAllConfidenceScores,
  }) async {
    final detectionResult = await _detector.detect(bitmap);

    if (detectionResult.isEmpty) {
      return OcrResult(
        boxes: [],
        texts: [],
        scores: [],
        characters: [],
        isRotated180: [],
      );
    }

    final croppedImages = <img.Image>[];
    final preparedBoxes = <List<Point>>[];
    for (final box in detectionResult) {
      final orderedPoints = _prepareRecognitionBox(bitmap, box.points);
      final cropped = ImageUtils.cropTextRegion(
        bitmap,
        orderedPoints,
        trimWhitespace: processingOptions.trimRecognitionWhitespace,
      );
      preparedBoxes.add(orderedPoints);
      croppedImages.add(cropped);
    }

    await _dumpPreparedCrops(detectionResult, preparedBoxes, croppedImages);

    final classificationMask = List<bool>.filled(croppedImages.length, false);
    final rotationStates = List<bool>.filled(croppedImages.length, false);

    if (useAngleClassification && _classifier != null) {
      final aspectCandidates = <int>[];
      for (int index = 0; index < croppedImages.length; index++) {
        final aspectRatio =
            croppedImages[index].width / croppedImages[index].height;
        if (aspectRatio < angleAspectRatioThreshold) {
          aspectCandidates.add(index);
        }
      }

      await classifyAndRotateIndices(
        croppedImages,
        aspectCandidates,
        classificationMask,
        rotationStates,
      );
    }

    var recognitionResults = await _recognizer.recognize(
      croppedImages,
      options: processingOptions,
    );

    if (useAngleClassification &&
        _classifier != null &&
        recognitionResults.isNotEmpty) {
      final lowConfidenceIndices = <int>[];
      for (int index = 0; index < recognitionResults.length; index++) {
        if (!classificationMask[index] &&
            recognitionResults[index].confidence < lowConfidenceThreshold) {
          lowConfidenceIndices.add(index);
        }
      }

      if (lowConfidenceIndices.isNotEmpty) {
        await classifyAndRotateIndices(
          croppedImages,
          lowConfidenceIndices,
          classificationMask,
          rotationStates,
        );

        final refreshed = await _recognizer.recognize(
          lowConfidenceIndices.map((i) => croppedImages[i]).toList(),
          options: processingOptions,
        );

        for (
          int refreshedIndex = 0;
          refreshedIndex < lowConfidenceIndices.length;
          refreshedIndex++
        ) {
          final originalIndex = lowConfidenceIndices[refreshedIndex];
          final current = recognitionResults[originalIndex];
          final updated = refreshed[refreshedIndex];
          if (updated.confidence > current.confidence) {
            recognitionResults[originalIndex] = updated;
          }
        }
      }
    }

    final characterBoxesPerDetection = <List<CharacterBox>>[];
    for (int index = 0; index < recognitionResults.length; index++) {
      characterBoxesPerDetection.add(
        buildCharacterBoxes(
          detectionResult[index],
          recognitionResults[index].characterSpans,
          rotationStates[index],
        ),
      );
    }

    final minThreshold = includeAllConfidenceScores
        ? fallbackMinRecognitionScore
        : minRecognitionScore;
    final filteredResults = <TextBox>[];
    final filteredTexts = <String>[];
    final filteredScores = <double>[];
    final filteredCharacters = <List<CharacterBox>>[];
    final filteredRotated180 = <bool>[];

    for (int i = 0; i < recognitionResults.length; i++) {
      final recognition = recognitionResults[i];
      if (recognition.confidence >= minThreshold) {
        filteredResults.add(detectionResult[i]);
        filteredTexts.add(recognition.text);
        filteredScores.add(recognition.confidence);
        filteredCharacters.add(characterBoxesPerDetection[i]);
        filteredRotated180.add(rotationStates[i]);
      }
    }

    debugPrint('Filtered to ${filteredResults.length} results');

    await debugSession?.writeJson('04_results/final_results.json', {
      'count': filteredResults.length,
      'results': List.generate(filteredResults.length, (index) {
        return {
          'text': filteredTexts[index],
          'confidence': filteredScores[index],
          'isRotated180': filteredRotated180[index],
          'points': filteredResults[index].points
              .map((point) => {'x': point.x, 'y': point.y})
              .toList(growable: false),
        };
      }),
    });

    return OcrResult(
      boxes: filteredResults,
      texts: filteredTexts,
      scores: filteredScores,
      characters: filteredCharacters,
      isRotated180: filteredRotated180,
    );
  }

  Future<_PageOrientationSelection> _selectBestPageOrientation(
    img.Image bitmap,
  ) async {
    _PageOrientationSelection? best;

    for (final angle in _pageOrientationProbeAngles) {
      final candidateBitmap = angle == 0
          ? bitmap
          : img.copyRotate(bitmap, angle: angle);
      final summary = await _detector.collectHighConfidenceDetections(
        candidateBitmap,
        minimumDetectionConfidence: minRecognitionScore,
        maxCandidates: quickCheckMaxCandidates,
      );
      final selection = _PageOrientationSelection(
        angle: angle,
        candidateCount: summary.candidates.length,
        maxDetectionScore: summary.maxDetectionScore ?? 0.0,
      );

      if (best == null || selection.compareTo(best) > 0) {
        best = selection;
      }
    }

    return best ?? const _PageOrientationSelection(angle: 0);
  }

  static OcrResult _mapResultToOriginalOrientation(
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
      isRotated180: List<bool>.from(result.isRotated180),
    );
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

  @visibleForTesting
  static OcrResult mapResultToOriginalOrientationForTest(
    OcrResult result, {
    required int angle,
    required int originalWidth,
    required int originalHeight,
  }) {
    return _mapResultToOriginalOrientation(
      result,
      angle: angle,
      originalWidth: originalWidth,
      originalHeight: originalHeight,
    );
  }

  Future<QuickCheckResult> hasHighConfidenceText(
    img.Image bitmap, {
    double minimumDetectionConfidence = 0.7,
    double recognitionThreshold = minRecognitionScore,
  }) async {
    final detectionSummary = await _detector.collectHighConfidenceDetections(
      bitmap,
      minimumDetectionConfidence: minimumDetectionConfidence,
      maxCandidates: quickCheckMaxCandidates,
    );

    if (detectionSummary.candidates.isEmpty) {
      return QuickCheckResult(
        hasText: false,
        detectorHit: false,
        examinedDetections: detectionSummary.examinedDetections,
        candidateCount: 0,
        evaluatedCandidates: 0,
        maxDetectionScore: detectionSummary.maxDetectionScore,
        bestRecognitionScore: null,
        bestRecognitionText: null,
        matchedDetectionScore: null,
      );
    }

    int evaluated = 0;
    bool matched = false;
    double? matchedDetectionScore;
    RecognitionResult? bestRecognition;
    var bestRecognitionScore = double.negativeInfinity;

    for (final candidate in detectionSummary.candidates) {
      evaluated++;
      final recognition = await recognizeCandidate(bitmap, candidate.box);
      if (recognition != null) {
        if (recognition.confidence > bestRecognitionScore) {
          bestRecognitionScore = recognition.confidence;
          bestRecognition = recognition;
        }
        final meetsThreshold =
            recognition.confidence >= recognitionThreshold &&
            recognition.text.isNotEmpty;
        if (meetsThreshold) {
          matched = true;
          matchedDetectionScore = candidate.score;
          break;
        }
      }
    }

    final bestScore = bestRecognitionScore == double.negativeInfinity
        ? null
        : bestRecognitionScore;
    return QuickCheckResult(
      hasText: matched,
      detectorHit: true,
      examinedDetections: detectionSummary.examinedDetections,
      candidateCount: detectionSummary.candidates.length,
      evaluatedCandidates: evaluated,
      maxDetectionScore: detectionSummary.maxDetectionScore,
      bestRecognitionScore: bestScore,
      bestRecognitionText: bestRecognition?.text,
      matchedDetectionScore: matchedDetectionScore,
    );
  }

  Future<RecognitionResult?> recognizeCandidate(
    img.Image bitmap,
    TextBox box,
  ) async {
    final orderedPoints = _prepareRecognitionBox(bitmap, box.points);
    final crop = ImageUtils.cropTextRegion(
      bitmap,
      orderedPoints,
      trimWhitespace: processingOptions.trimRecognitionWhitespace,
    );
    final crops = [crop];
    final classificationMask = [false];
    final rotationStates = [false];

    if (useAngleClassification && _classifier != null) {
      final aspectRatio = crop.width / crop.height;
      if (aspectRatio < angleAspectRatioThreshold) {
        await classifyAndRotateIndices(
          crops,
          [0],
          classificationMask,
          rotationStates,
        );
      }
    }

    var recognitionResults = await _recognizer.recognize(
      crops,
      options: processingOptions,
    );

    if (useAngleClassification &&
        _classifier != null &&
        recognitionResults.isNotEmpty) {
      final needsRetry =
          !classificationMask[0] &&
          recognitionResults[0].confidence < lowConfidenceThreshold;
      if (needsRetry) {
        await classifyAndRotateIndices(
          crops,
          [0],
          classificationMask,
          rotationStates,
        );
        final refreshed = await _recognizer.recognize(
          crops,
          options: processingOptions,
        );
        if (refreshed.isNotEmpty &&
            refreshed[0].confidence > recognitionResults[0].confidence) {
          recognitionResults = refreshed;
        }
      }
    }

    return recognitionResults.firstOrNull;
  }

  Future<void> classifyAndRotateIndices(
    List<img.Image> images,
    List<int> indices,
    List<bool> classificationMask,
    List<bool> rotationStates,
  ) async {
    if (!useAngleClassification || _classifier == null || indices.isEmpty) {
      return;
    }

    final subset = indices.map((i) => images[i]).toList();
    final outputs = await _classifier!.classifyAndRotate(subset);

    for (int idx = 0; idx < indices.length; idx++) {
      final imageIndex = indices[idx];
      classificationMask[imageIndex] = true;
      final output = outputs[idx];
      if (output.rotated) {
        rotationStates[imageIndex] = !rotationStates[imageIndex];
      }
      images[imageIndex] = output.bitmap;
    }
  }

  List<CharacterBox> buildCharacterBoxes(
    TextBox textBox,
    List<CharacterSpan> spans,
    bool rotated,
  ) {
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

          final topStart = interpolate(topLeft, topRight, clampedStart);
          final topEnd = interpolate(topLeft, topRight, clampedEnd);
          final bottomStart = interpolate(
            bottomLeft,
            bottomRight,
            clampedStart,
          );
          final bottomEnd = interpolate(bottomLeft, bottomRight, clampedEnd);

          return CharacterBox(
            text: span.text,
            confidence: span.confidence,
            points: [topStart, topEnd, bottomEnd, bottomStart],
          );
        })
        .whereType<CharacterBox>()
        .toList();
  }

  Point interpolate(Point start, Point end, double ratio) {
    final clamped = ratio.clamp(0.0, 1.0);
    return Point(
      start.x + (end.x - start.x) * clamped,
      start.y + (end.y - start.y) * clamped,
    );
  }

  List<Point> _prepareRecognitionBox(img.Image bitmap, List<Point> points) {
    return ImageUtils.orderPointsClockwise(
      ImageUtils.clipBoxToImageBounds(points, bitmap.width, bitmap.height),
    );
  }

  Future<void> _dumpPreparedCrops(
    List<TextBox> detectionResult,
    List<List<Point>> preparedBoxes,
    List<img.Image> croppedImages,
  ) async {
    if (debugSession == null) {
      return;
    }

    final boxMetadata = <Map<String, Object?>>[];
    for (int index = 0; index < croppedImages.length; index++) {
      final rawBox = detectionResult[index].points;
      final preparedBox = preparedBoxes[index];
      final crop = croppedImages[index];
      final prefix = index.toString().padLeft(3, '0');
      await debugSession!.saveImage('02_crops/${prefix}_crop.png', crop);
      boxMetadata.add({
        'index': index,
        'rawPoints': rawBox
            .map((point) => {'x': point.x, 'y': point.y})
            .toList(growable: false),
        'preparedPoints': preparedBox
            .map((point) => {'x': point.x, 'y': point.y})
            .toList(growable: false),
        'cropWidth': crop.width,
        'cropHeight': crop.height,
      });
    }

    await debugSession!.writeJson('02_crops/boxes.json', boxMetadata);
  }

  Future<void> close() async {
    detectionSession.release();
    recognitionSession.release();
    classificationSession?.release();
    OrtEnv.instance.release();
  }
}

class _PageOrientationSelection
    implements Comparable<_PageOrientationSelection> {
  final int angle;
  final int candidateCount;
  final double maxDetectionScore;

  const _PageOrientationSelection({
    required this.angle,
    this.candidateCount = 0,
    this.maxDetectionScore = 0.0,
  });

  @override
  int compareTo(_PageOrientationSelection other) {
    final candidateComparison = candidateCount.compareTo(other.candidateCount);
    if (candidateComparison != 0) {
      return candidateComparison;
    }

    final scoreComparison = maxDetectionScore.compareTo(
      other.maxDetectionScore,
    );
    if (scoreComparison != 0) {
      return scoreComparison;
    }

    if (angle == 0 && other.angle != 0) {
      return 1;
    }
    if (angle != 0 && other.angle == 0) {
      return -1;
    }
    return 0;
  }
}
