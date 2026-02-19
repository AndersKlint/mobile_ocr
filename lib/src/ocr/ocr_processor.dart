import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';
import 'package:image/image.dart' as img;
import 'types.dart';
import 'image_utils.dart';
import 'text_detector.dart';
import 'text_recognizer.dart';
import 'text_classifier.dart';

class OcrProcessor {
  static const double minRecognitionScore = 0.8;
  static const double fallbackMinRecognitionScore = 0.5;
  static const double angleAspectRatioThreshold = 0.5;
  static const double lowConfidenceThreshold = 0.65;
  static const int quickCheckMaxCandidates = 3;

  final OrtSession detectionSession;
  final OrtSession recognitionSession;
  final OrtSession? classificationSession;
  final List<String> characterDict;
  final bool useAngleClassification;

  late final TextDetector _detector;
  late final TextRecognizer _recognizer;
  TextClassifier? _classifier;

  OcrProcessor({
    required this.detectionSession,
    required this.recognitionSession,
    required this.classificationSession,
    required this.characterDict,
    this.useAngleClassification = true,
  }) {
    _detector = TextDetector(detectionSession);
    _recognizer = TextRecognizer(recognitionSession, characterDict);
    if (useAngleClassification && classificationSession != null) {
      _classifier = TextClassifier(classificationSession!);
    }
  }

  static Future<OcrProcessor> create({
    required String detectionModelPath,
    required String recognitionModelPath,
    required String? classificationModelPath,
    required String dictionaryPath,
    bool useAngleClassification = true,
  }) async {
    final ort = OnnxRuntime();

    final detectionSession = await ort.createSession(detectionModelPath);
    final recognitionSession = await ort.createSession(recognitionModelPath);
    OrtSession? classificationSession;
    if (useAngleClassification && classificationModelPath != null) {
      classificationSession = await ort.createSession(classificationModelPath);
    }

    final dictFile = File(dictionaryPath);
    final dictLines = await dictFile.readAsLines();
    final characterDict = ['blank', ...dictLines, ' '];

    return OcrProcessor(
      detectionSession: detectionSession,
      recognitionSession: recognitionSession,
      classificationSession: classificationSession,
      characterDict: characterDict,
      useAngleClassification: useAngleClassification,
    );
  }

  Future<OcrResult> processImage(
    img.Image bitmap, {
    bool includeAllConfidenceScores = false,
  }) async {
    final detectionResult = await _detector.detect(bitmap);

    if (detectionResult.isEmpty) {
      return OcrResult(boxes: [], texts: [], scores: [], characters: []);
    }

    final croppedImages = <img.Image>[];
    for (final box in detectionResult) {
      final orderedPoints = ImageUtils.orderPointsClockwise(box.points);
      final cropped = ImageUtils.cropTextRegion(bitmap, orderedPoints);
      croppedImages.add(cropped);
    }

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

    var recognitionResults = await _recognizer.recognize(croppedImages);

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

    for (int i = 0; i < recognitionResults.length; i++) {
      final recognition = recognitionResults[i];
      if (recognition.confidence >= minThreshold) {
        filteredResults.add(detectionResult[i]);
        filteredTexts.add(recognition.text);
        filteredScores.add(recognition.confidence);
        filteredCharacters.add(characterBoxesPerDetection[i]);
      }
    }

    debugPrint('Filtered to ${filteredResults.length} results');

    return OcrResult(
      boxes: filteredResults,
      texts: filteredTexts,
      scores: filteredScores,
      characters: filteredCharacters,
    );
  }

  Future<QuickCheckResult> hasHighConfidenceText(
    img.Image bitmap, {
    double minimumDetectionConfidence = 0.9,
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
    final orderedPoints = ImageUtils.orderPointsClockwise(box.points);
    final crop = ImageUtils.cropTextRegion(bitmap, orderedPoints);
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

    var recognitionResults = await _recognizer.recognize(crops);

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
        final refreshed = await _recognizer.recognize(crops);
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

  Future<void> close() async {
    await detectionSession.close();
    await recognitionSession.close();
    await classificationSession?.close();
  }
}
