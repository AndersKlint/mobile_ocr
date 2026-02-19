import 'dart:typed_data';
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';
import 'package:image/image.dart' as img;
import 'types.dart';

class TextRecognizer {
  static const int imgHeight = 48;
  static const int imgWidth = 320;
  static const int batchSize = 6;
  static const double minSpanRatio = 1e-3;

  final OrtSession session;
  final List<String> characterDict;

  TextRecognizer(this.session, this.characterDict);

  Future<List<RecognitionResult>> recognize(List<img.Image> images) async {
    if (images.isEmpty) {
      return [];
    }

    final widthList = images.map((img) => img.width / img.height).toList();
    final sortedIndices = List.generate(widthList.length, (i) => i)
      ..sort((a, b) => widthList[a].compareTo(widthList[b]));

    final orderedResults = List<RecognitionResult>.filled(
      images.length,
      RecognitionResult(text: '', confidence: 0, characterSpans: []),
    );

    for (int start = 0; start < sortedIndices.length; start += batchSize) {
      final end = (start + batchSize).clamp(0, sortedIndices.length);
      final batchIndices = sortedIndices.sublist(start, end);
      final batchBitmaps = batchIndices.map((i) => images[i]).toList();
      final batchResults = await processBatch(batchBitmaps);

      for (int idx = 0; idx < batchIndices.length; idx++) {
        orderedResults[batchIndices[idx]] = batchResults[idx];
      }
    }

    return orderedResults;
  }

  Future<List<RecognitionResult>> processBatch(
    List<img.Image> batchImages,
  ) async {
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

    final contentWidths = List<int>.filled(batchSz, 0);
    for (int index = 0; index < batchImages.length; index++) {
      contentWidths[index] = preprocessImage(
        batchImages[index],
        inputArray,
        index,
        targetWidth,
      );
    }

    final shape = [batchSz, 3, imgHeight, targetWidth];
    final inputTensor = await OrtValue.fromList(inputArray, shape);

    final inputs = {session.inputNames.first: inputTensor};
    final outputs = await session.run(inputs);
    final output = outputs.values.first;

    final results = await decodeOutput(
      output,
      batchSz,
      contentWidths,
      targetWidth,
    );

    await inputTensor.dispose();
    await output.dispose();

    return results;
  }

  int preprocessImage(
    img.Image bitmap,
    Float32List outputArray,
    int batchIndex,
    int targetWidth,
  ) {
    final aspectRatio = bitmap.width / bitmap.height;
    final resizedWidth = (imgHeight * aspectRatio).ceil().clamp(1, targetWidth);

    final resized = img.copyResize(
      bitmap,
      width: resizedWidth,
      height: imgHeight,
      interpolation: img.Interpolation.linear,
    );

    final baseOffset = batchIndex * 3 * imgHeight * targetWidth;
    final channelStride = imgHeight * targetWidth;

    for (int y = 0; y < imgHeight; y++) {
      final rowOffset = y * targetWidth;

      for (int x = 0; x < targetWidth; x++) {
        final pixelIndex = rowOffset + x;

        if (x < resizedWidth) {
          final pixel = resized.getPixel(x, y);
          final r = pixel.r.toDouble() / 255.0;
          final g = pixel.g.toDouble() / 255.0;
          final b = pixel.b.toDouble() / 255.0;

          // BGR order to match Kotlin/Android
          outputArray[baseOffset + pixelIndex] = (b - 0.5) / 0.5;
          outputArray[baseOffset + channelStride + pixelIndex] =
              (g - 0.5) / 0.5;
          outputArray[baseOffset + 2 * channelStride + pixelIndex] =
              (r - 0.5) / 0.5;
        } else {
          outputArray[baseOffset + pixelIndex] = 0;
          outputArray[baseOffset + channelStride + pixelIndex] = 0;
          outputArray[baseOffset + 2 * channelStride + pixelIndex] = 0;
        }
      }
    }

    return resizedWidth;
  }

  Future<List<RecognitionResult>> decodeOutput(
    OrtValue output,
    int batchSz,
    List<int> contentWidths,
    int targetWidth,
  ) async {
    final outputData = await output.asFlattenedList();
    final shape = output.shape;
    final seqLen = shape[1];
    final vocabSize = shape[2];

    final results = <RecognitionResult>[];

    for (int b = 0; b < batchSz; b++) {
      final batchOffset = b * seqLen * vocabSize;

      final charIndices = List<int>.filled(seqLen, 0);
      final probs = List<double>.filled(seqLen, 0);

      for (int t = 0; t < seqLen; t++) {
        final timeOffset = batchOffset + t * vocabSize;

        var maxProb = (outputData[timeOffset] as num).toDouble();
        var maxIndex = 0;

        for (int c = 1; c < vocabSize; c++) {
          final idx = timeOffset + c;
          if (idx >= outputData.length) {
            break;
          }
          final prob = (outputData[idx] as num).toDouble();
          if (prob > maxProb) {
            maxProb = prob;
            maxIndex = c;
          }
        }

        charIndices[t] = maxIndex;
        probs[t] = maxProb;
      }

      final contentWidth = contentWidths[b] > 0
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
    List<int> charIndices,
    List<double> probs,
    double scale,
  ) {
    final seqLen = charIndices.length;
    if (seqLen == 0) {
      return RecognitionResult(text: '', confidence: 0, characterSpans: []);
    }

    final safeScale = scale.isFinite && scale > 0 ? scale : 1.0;
    final decodedChars = <String>[];
    final decodedProbs = <double>[];
    final spans = <CharacterSpan>[];

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

      while (end < seqLen && charIndices[end] == currentIndex) {
        probSum += probs[end];
        end++;
        count++;
      }

      if (currentIndex < characterDict.length) {
        final character = characterDict[currentIndex];
        decodedChars.add(character);

        final meanProb = probSum / count;
        decodedProbs.add(meanProb);

        final minSpan = ((1 / seqLen) * safeScale).clamp(minSpanRatio, 1.0);

        var startRatio = (start / seqLen) * safeScale;
        var endRatio = (end / seqLen) * safeScale;

        startRatio = startRatio.clamp(0.0, 1.0);
        endRatio = endRatio.clamp(startRatio, 1.0);

        if (endRatio - startRatio < minSpan) {
          endRatio = (startRatio + minSpan).clamp(0.0, 1.0);
          if (endRatio - startRatio < minSpan) {
            startRatio = (endRatio - minSpan).clamp(0.0, 1.0);
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

    final text = decodedChars.join();
    final confidence = decodedProbs.isNotEmpty
        ? decodedProbs.reduce((a, b) => a + b) / decodedProbs.length
        : 0.0;

    return RecognitionResult(
      text: text,
      confidence: confidence,
      characterSpans: spans,
    );
  }
}
