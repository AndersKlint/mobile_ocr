import 'dart:typed_data';
import 'package:onnxruntime_v2/onnxruntime_v2.dart';
import 'package:image/image.dart' as img;
import 'types.dart';
import 'fast_image_loader.dart';
import 'fast_tensor_reader.dart';
import 'image_utils.dart';
import 'ocr_debug_dumper.dart';

class TextRecognizer {
  static const int imgHeight = 48;
  static const int imgWidth = 320;
  static const int batchSize = 6;
  static const double minSpanRatio = 1e-3;
  static const List<double> _mean = [0.5, 0.5, 0.5];
  static const List<double> _std = [0.5, 0.5, 0.5];

  final OrtSession session;
  final List<String> characterDict;
  final OcrDebugSession? debugSession;

  TextRecognizer(this.session, this.characterDict, {this.debugSession});

  Future<List<RecognitionResult>> recognize(
    List<img.Image> images, {
    OcrProcessingOptions options = const OcrProcessingOptions(),
  }) async {
    if (images.isEmpty) {
      return [];
    }

    final widthList = images.map((img) => img.width / img.height).toList();
    final sortedIndices = List.generate(widthList.length, (i) => i)
      ..sort((a, b) => widthList[a].compareTo(widthList[b]));
    final resizedWidths = sortedIndices
        .map((index) => (imgHeight * widthList[index]).ceil().clamp(1, 10000))
        .toList(growable: false);

    final orderedResults = List<RecognitionResult>.filled(
      images.length,
      RecognitionResult(text: '', confidence: 0, characterSpans: []),
    );

    int start = 0;
    while (start < sortedIndices.length) {
      var end = start + 1;
      var widest = resizedWidths[start];

      while (end < sortedIndices.length && end - start < batchSize) {
        final candidateWidth = resizedWidths[end];
        final nextWidest = candidateWidth > widest ? candidateWidth : widest;
        final nextBatchSize = end - start + 1;
        final exceedsPixelBudget =
            nextBatchSize * nextWidest > batchSize * imgWidth;
        if (exceedsPixelBudget) {
          break;
        }
        widest = nextWidest;
        end++;
      }

      final batchIndices = sortedIndices.sublist(start, end);
      final batchBitmaps = batchIndices.map((i) => images[i]).toList();
      final batchResults = await processBatch(
        batchBitmaps,
        batchStartIndex: start,
        options: options,
      );

      for (int idx = 0; idx < batchIndices.length; idx++) {
        orderedResults[batchIndices[idx]] = batchResults[idx];
      }

      start = end;
    }

    return orderedResults;
  }

  Future<List<RecognitionResult>> processBatch(
    List<img.Image> batchImages, {
    int batchStartIndex = 0,
    OcrProcessingOptions options = const OcrProcessingOptions(),
  }) async {
    if (batchImages.isEmpty) return [];

    var maxWhRatio = imgWidth / imgHeight.toDouble();
    for (final image in batchImages) {
      final ratio = image.width / image.height;
      if (ratio > maxWhRatio) {
        maxWhRatio = ratio;
      }
    }

    final targetWidth = (imgHeight * maxWhRatio).ceil().clamp(1, 10000);

    final batchSz = batchImages.length;
    final inputArray = Float32List(batchSz * 3 * imgHeight * targetWidth);
    inputArray.fillRange(0, inputArray.length, 0);

    final contentWidths = List<int>.filled(batchSz, 0);
    for (int index = 0; index < batchImages.length; index++) {
      contentWidths[index] = await preprocessImage(
        batchImages[index],
        inputArray,
        index,
        targetWidth,
        batchStartIndex: batchStartIndex + index,
        options: options,
      );
    }

    final shape = [batchSz, 3, imgHeight, targetWidth];
    final inputTensor = OrtValueTensor.createTensorWithDataList(
      inputArray,
      shape,
    );

    final inputs = {session.inputNames.first: inputTensor};
    final runOptions = OrtRunOptions();
    final outputs = session.run(runOptions, inputs);
    runOptions.release();
    final output = outputs[0];

    List<RecognitionResult> results = [];
    if (output != null) {
      results = decodeOutput(output, batchSz, contentWidths, targetWidth);
      output.release();
    }

    inputTensor.release();

    return results;
  }

  Future<int> preprocessImage(
    img.Image bitmap,
    Float32List outputArray,
    int batchIndex,
    int targetWidth, {
    int batchStartIndex = 0,
    OcrProcessingOptions options = const OcrProcessingOptions(),
  }) async {
    final debugBatchIndex = batchStartIndex + batchIndex;
    final preparedBitmap = options.enhanceRecognitionCrops
        ? ImageUtils.enhanceRecognitionCrop(
            bitmap,
            contrastBoost: options.recognitionContrastBoost,
            brightnessBoost: options.recognitionBrightnessBoost,
          )
        : bitmap;
    final aspectRatio = bitmap.width / bitmap.height;
    final resizedWidth = (imgHeight * aspectRatio).ceil().clamp(1, targetWidth);
    final baseOffset = batchIndex * 3 * imgHeight * targetWidth;

    final wroteTensor = await FastImageLoader.resizeToNormalizedBatchBuffer(
      preparedBitmap,
      resizedWidth: resizedWidth,
      targetWidth: targetWidth,
      targetHeight: imgHeight,
      buffer: outputArray,
      bufferOffset: baseOffset,
      mean: _mean,
      std: _std,
      bgrOrder: true,
    );
    if (!wroteTensor) {
      return 0;
    }

    if (debugSession != null) {
      final resizedPreview = img.copyResize(
        preparedBitmap,
        width: resizedWidth,
        height: imgHeight,
        interpolation: img.Interpolation.linear,
      );
      final modelInputPreview = buildModelInputPreview(
        resizedImage: resizedPreview,
        targetWidth: targetWidth,
        targetHeight: imgHeight,
      );
      await debugSession!.saveImage(
        '03_recognizer/${debugBatchIndex.toString().padLeft(3, '0')}_crop.png',
        bitmap,
      );
      await debugSession!.saveImage(
        '03_recognizer/${debugBatchIndex.toString().padLeft(3, '0')}_model_input.png',
        modelInputPreview,
      );
      await debugSession!.writeJson(
        '03_recognizer/${debugBatchIndex.toString().padLeft(3, '0')}_meta.json',
        {
          'originalWidth': bitmap.width,
          'originalHeight': bitmap.height,
          'enhanceRecognitionCrops': options.enhanceRecognitionCrops,
          'contrastBoost': options.enhanceRecognitionCrops
              ? options.recognitionContrastBoost
              : 0.0,
          'brightnessBoost': options.enhanceRecognitionCrops
              ? options.recognitionBrightnessBoost
              : 0.0,
          'resizedWidth': resizedWidth,
          'targetWidth': targetWidth,
          'targetHeight': imgHeight,
          'aspectRatio': aspectRatio,
        },
      );
    }

    return resizedWidth;
  }

  List<RecognitionResult> decodeOutput(
    OrtValue output,
    int batchSz,
    List<int> contentWidths,
    int targetWidth,
  ) {
    final flatData = FastTensorReader.asFloat32List(output);
    if (flatData == null || flatData.isEmpty) return [];

    final shape = FastTensorReader.getShape(output);
    if (shape.length < 3) return [];

    final seqLen = shape[1];
    final vocabSize = shape[2];
    if (seqLen == 0 || vocabSize == 0) return [];

    final results = <RecognitionResult>[];
    final seqStride = seqLen * vocabSize;
    final vocabStride = vocabSize;

    for (int b = 0; b < batchSz; b++) {
      final batchOffset = b * seqStride;

      final charIndices = Int32List(seqLen);
      final probs = Float32List(seqLen);

      for (int t = 0; t < seqLen; t++) {
        final timeOffset = batchOffset + t * vocabStride;

        var maxProb = flatData[timeOffset];
        var maxIndex = 0;

        for (int c = 1; c < vocabSize; c++) {
          final prob = flatData[timeOffset + c];
          if (prob > maxProb) {
            maxProb = prob;
            maxIndex = c;
          }
        }

        charIndices[t] = maxIndex;
        probs[t] = maxProb;
      }

      final contentWidth = b < contentWidths.length && contentWidths[b] > 0
          ? contentWidths[b]
          : targetWidth;
      final scaleFactor = contentWidth >= targetWidth
          ? 1.0
          : targetWidth / contentWidth;
      final recognition = ctcDecode(charIndices, probs, scaleFactor);

      results.add(recognition);
    }

    return results;
  }

  RecognitionResult ctcDecode(
    Int32List charIndices,
    Float32List probs,
    double scale,
  ) {
    final seqLen = charIndices.length;
    if (seqLen == 0) {
      return RecognitionResult(text: '', confidence: 0, characterSpans: []);
    }

    final safeScale = scale.isFinite && scale > 0 ? scale : 1.0;
    final textBuffer = StringBuffer();
    final spans = <CharacterSpan>[];
    var confidenceSum = 0.0;
    var confidenceCount = 0;

    final invSeqLen = 1.0 / seqLen;
    final dictLength = characterDict.length;

    int t = 0;
    while (t < seqLen) {
      final currentIndex = charIndices[t];

      if (currentIndex == 0) {
        t++;
        continue;
      }

      final start = t;
      var end = t + 1;
      var probSum = probs[t];
      var count = 1;

      final currentIdx = currentIndex;
      while (end < seqLen && charIndices[end] == currentIdx) {
        probSum += probs[end];
        end++;
        count++;
      }

      if (currentIndex < dictLength) {
        final character = characterDict[currentIndex];
        textBuffer.write(character);

        final meanProb = probSum / count;
        confidenceSum += meanProb;
        confidenceCount++;

        final minSpan = (invSeqLen * safeScale).clamp(minSpanRatio, 1.0);

        var startRatio = (start * invSeqLen) * safeScale;
        var endRatio = (end * invSeqLen) * safeScale;

        if (startRatio < 0.0) startRatio = 0.0;
        if (endRatio < startRatio) endRatio = startRatio;
        if (startRatio > 1.0) startRatio = 1.0;
        if (endRatio > 1.0) endRatio = 1.0;

        if (endRatio - startRatio < minSpan) {
          endRatio = startRatio + minSpan;
          if (endRatio > 1.0) {
            endRatio = 1.0;
            startRatio = endRatio - minSpan;
            if (startRatio < 0.0) startRatio = 0.0;
          }
        }

        spans.add(
          CharacterSpan(
            text: character,
            confidence: meanProb,
            startRatio: startRatio,
            endRatio: endRatio,
          ),
        );
      }

      t = end;
    }

    final text = textBuffer.toString();
    final confidence = confidenceCount > 0
        ? confidenceSum / confidenceCount
        : 0.0;

    return RecognitionResult(
      text: text,
      confidence: confidence,
      characterSpans: spans,
    );
  }
}
