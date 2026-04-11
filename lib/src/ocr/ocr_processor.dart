import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:image/image.dart' as img;
import 'package:onnxruntime_v2/onnxruntime_v2.dart';

import 'image_utils.dart';
import 'ocr_character_box_builder.dart';
import 'ocr_debug_dumper.dart';
import 'ocr_orientation_mapper.dart';
import 'text_classifier.dart';
import 'text_detector.dart';
import 'text_recognizer.dart';
import 'types.dart';

class OcrProcessor {
  static const double minRecognitionScore = 0.7;
  static const double fallbackMinRecognitionScore = 0.5;
  static const double angleAspectRatioThreshold = 0.5;
  static const double lowConfidenceThreshold = 0.55;
  static const int quickCheckMaxCandidates = 3;
  static const List<int> _pageOrientationProbeAngles = [0, 90];

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
    final normalizedAngle = selectedOrientation.angle;
    final workingBitmap = selectedOrientation.bitmap;

    await debugSession?.writeJson('00_source/page_orientation.json', {
      'selectedAngle': selectedOrientation.angle,
      'normalizedAngle': normalizedAngle,
      'isLandscape': selectedOrientation.isLandscape,
      'candidateCount': selectedOrientation.candidateCount,
      'maxDetectionScore': selectedOrientation.maxDetectionScore,
    });

    final preferLandscapeRecognitionBoxes =
        normalizedAngle == 90 || normalizedAngle == 270;
    final rawResult = await _processOrientedImage(
      workingBitmap,
      includeAllConfidenceScores: includeAllConfidenceScores,
      preferLandscapeRecognitionBoxes: preferLandscapeRecognitionBoxes,
      classifyAllRecognitionCrops: preferLandscapeRecognitionBoxes,
    );

    if (normalizedAngle == 0 || rawResult.boxes.isEmpty) {
      return rawResult;
    }

    return OcrOrientationMapper.mapResultToOriginalOrientation(
      rawResult,
      angle: normalizedAngle,
      originalWidth: bitmap.width,
      originalHeight: bitmap.height,
    );
  }

  Future<OcrResult> _processOrientedImage(
    img.Image bitmap, {
    required bool includeAllConfidenceScores,
    bool preferLandscapeRecognitionBoxes = false,
    bool classifyAllRecognitionCrops = false,
  }) async {
    final detectionResult = await _detector.detect(bitmap);

    if (detectionResult.isEmpty) {
      return OcrResult(
        boxes: [],
        texts: [],
        scores: [],
        characters: [],
        textOrientations: [],
      );
    }

    final preparedRecognition = _prepareRecognitionCrops(
      bitmap,
      detectionResult,
      preferLandscapeFallback: preferLandscapeRecognitionBoxes,
    );
    final preparedBoxes = preparedRecognition.preparedBoxes;
    final croppedImages = preparedRecognition.croppedImages;
    final dominantLandscapePreference = preparedRecognition.preferLandscape;

    await _dumpPreparedCrops(detectionResult, preparedBoxes, croppedImages);

    final classificationMask = List<bool>.filled(croppedImages.length, false);
    final rotationStates = List<bool>.filled(croppedImages.length, false);

    if (useAngleClassification && _classifier != null) {
      final aspectCandidates = <int>[];
      for (int index = 0; index < croppedImages.length; index++) {
        if (classifyAllRecognitionCrops) {
          aspectCandidates.add(index);
          continue;
        }
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

    final defaultRotated180 = _resolveDefaultRotated180(
      classificationMask: classificationMask,
      rotationStates: rotationStates,
    );

    final characterBoxesPerDetection = <List<CharacterBox>>[];
    final textOrientationsPerDetection = <String>[];
    for (int index = 0; index < recognitionResults.length; index++) {
      final rotated180 = classificationMask[index]
          ? rotationStates[index]
          : defaultRotated180;
      textOrientationsPerDetection.add(
        OcrOrientationMapper.resolveTextOrientation(
          detectionResult[index],
          rotated180,
          preferLandscape: dominantLandscapePreference,
        ),
      );
      characterBoxesPerDetection.add(
        OcrCharacterBoxBuilder.build(
          detectionResult[index],
          recognitionResults[index].characterSpans,
          rotated180,
          preferLandscape: dominantLandscapePreference,
        ),
      );
      rotationStates[index] = rotated180;
    }

    final minThreshold = includeAllConfidenceScores
        ? fallbackMinRecognitionScore
        : minRecognitionScore;
    final filteredResults = <TextBox>[];
    final filteredTexts = <String>[];
    final filteredScores = <double>[];
    final filteredCharacters = <List<CharacterBox>>[];
    final filteredTextOrientations = <String>[];

    for (int i = 0; i < recognitionResults.length; i++) {
      final recognition = recognitionResults[i];
      if (recognition.confidence >= minThreshold) {
        filteredResults.add(detectionResult[i]);
        filteredTexts.add(recognition.text);
        filteredScores.add(recognition.confidence);
        filteredCharacters.add(characterBoxesPerDetection[i]);
        filteredTextOrientations.add(textOrientationsPerDetection[i]);
      }
    }

    debugPrint('Filtered to ${filteredResults.length} results');

    await debugSession?.writeJson('04_results/final_results.json', {
      'count': filteredResults.length,
      'results': List.generate(filteredResults.length, (index) {
        return {
          'text': filteredTexts[index],
          'confidence': filteredScores[index],
          'textOrientation': filteredTextOrientations[index],
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
      textOrientations: filteredTextOrientations,
    );
  }

  Future<_PageOrientationSelection> _selectBestPageOrientation(
    img.Image bitmap,
  ) async {
    _PageOrientationSelection? best;
    final bitmapIsLandscape = bitmap.width > bitmap.height;

    for (final angle in _pageOrientationProbeAngles) {
      final candidateBitmap = angle == 0
          ? bitmap
          : ImageUtils.rotateOrthogonal(bitmap, angle: angle);
      final summary = await _detector.collectHighConfidenceDetections(
        candidateBitmap,
        minimumDetectionConfidence: minRecognitionScore,
        maxCandidates: quickCheckMaxCandidates,
      );
      final selection = _PageOrientationSelection(
        bitmap: candidateBitmap,
        angle: angle,
        isLandscape: candidateBitmap.width > candidateBitmap.height,
        candidateCount: summary.candidates.length,
        maxDetectionScore: summary.maxDetectionScore ?? 0.0,
      );

      if (best == null || selection.compareTo(best) > 0) {
        best = selection;
      }
    }

    if (best == null) {
      return _PageOrientationSelection(
        bitmap: bitmap,
        angle: 0,
        isLandscape: false,
      );
    }

    if (best.isLandscape == bitmapIsLandscape) {
      return _PageOrientationSelection(
        bitmap: bitmap,
        angle: 0,
        isLandscape: bitmapIsLandscape,
        candidateCount: best.candidateCount,
        maxDetectionScore: best.maxDetectionScore,
      );
    }

    return best;
  }

  @visibleForTesting
  static OcrResult mapResultToOriginalOrientationForTest(
    OcrResult result, {
    required int angle,
    required int originalWidth,
    required int originalHeight,
  }) {
    return OcrOrientationMapper.mapResultToOriginalOrientation(
      result,
      angle: angle,
      originalWidth: originalWidth,
      originalHeight: originalHeight,
    );
  }

  @visibleForTesting
  static String resolveTextOrientationForTest(
    TextBox textBox,
    bool rotated180, {
    required bool preferLandscape,
  }) {
    return OcrOrientationMapper.resolveTextOrientation(
      textBox,
      rotated180,
      preferLandscape: preferLandscape,
    );
  }

  @visibleForTesting
  static String rotateTextOrientationForTest(
    String orientation, {
    required int angle,
  }) {
    return OcrOrientationMapper.rotateTextOrientation(
      orientation,
      angle: angle,
    );
  }

  @visibleForTesting
  static int normalizeOrientationAngleForTest({
    required bool sourceIsLandscape,
    required bool detectedIsLandscape,
  }) {
    return OcrOrientationMapper.normalizeOrientationAngle(
      sourceIsLandscape: sourceIsLandscape,
      detectedIsLandscape: detectedIsLandscape,
    );
  }

  @visibleForTesting
  static bool resolveDefaultRotated180ForTest({
    required List<bool> classificationMask,
    required List<bool> rotationStates,
  }) {
    return _resolveDefaultRotated180(
      classificationMask: classificationMask,
      rotationStates: rotationStates,
    );
  }

  @visibleForTesting
  static bool resolveRecognitionBoxLandscapePreferenceForTest(
    List<List<Point>> boxes, {
    required bool fallback,
  }) {
    return _resolveRecognitionBoxLandscapePreference(boxes, fallback: fallback);
  }

  static bool _resolveDefaultRotated180({
    required List<bool> classificationMask,
    required List<bool> rotationStates,
  }) {
    return OcrOrientationMapper.resolveDefaultRotated180(
      classificationMask: classificationMask,
      rotationStates: rotationStates,
    );
  }

  static bool _resolveRecognitionBoxLandscapePreference(
    List<List<Point>> boxes, {
    required bool fallback,
  }) {
    return OcrOrientationMapper.resolveRecognitionBoxLandscapePreference(
      boxes,
      fallback: fallback,
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

  _PreparedRecognitionCrops _prepareRecognitionCrops(
    img.Image bitmap,
    List<TextBox> boxes, {
    required bool preferLandscapeFallback,
  }) {
    final preparedBoxes = <List<Point>>[];
    for (final box in boxes) {
      preparedBoxes.add(_prepareRecognitionBox(bitmap, box.points));
    }

    final preferLandscape = _resolveRecognitionBoxLandscapePreference(
      preparedBoxes,
      fallback: preferLandscapeFallback,
    );
    final croppedImages = preparedBoxes
        .map(
          (points) => ImageUtils.cropTextRegion(
            bitmap,
            points,
            trimWhitespace: processingOptions.trimRecognitionWhitespace,
            preferLandscape: preferLandscape,
          ),
        )
        .toList(growable: false);

    return _PreparedRecognitionCrops(
      preparedBoxes: preparedBoxes,
      croppedImages: croppedImages,
      preferLandscape: preferLandscape,
    );
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
    bool rotated, {
    bool preferLandscape = false,
  }) {
    return OcrCharacterBoxBuilder.build(
      textBox,
      spans,
      rotated,
      preferLandscape: preferLandscape,
    );
  }

  @visibleForTesting
  static List<CharacterBox> buildCharacterBoxesForTest(
    TextBox textBox,
    List<CharacterSpan> spans,
    bool rotated, {
    bool preferLandscape = false,
  }) {
    return OcrCharacterBoxBuilder.build(
      textBox,
      spans,
      rotated,
      preferLandscape: preferLandscape,
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
  final img.Image bitmap;
  final int angle;
  final bool isLandscape;
  final int candidateCount;
  final double maxDetectionScore;

  _PageOrientationSelection({
    required this.bitmap,
    required this.angle,
    required this.isLandscape,
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

class _PreparedRecognitionCrops {
  final List<List<Point>> preparedBoxes;
  final List<img.Image> croppedImages;
  final bool preferLandscape;

  const _PreparedRecognitionCrops({
    required this.preparedBoxes,
    required this.croppedImages,
    required this.preferLandscape,
  });
}
