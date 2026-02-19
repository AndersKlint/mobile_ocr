import 'dart:typed_data';
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';
import 'package:image/image.dart' as img;

class ClassificationOutput {
  final img.Image bitmap;
  final bool rotated;

  ClassificationOutput(this.bitmap, this.rotated);
}

class TextClassifier {
  static const int imgHeight = 48;
  static const int imgWidth = 192;
  static const double clsThresh = 0.9;
  static const int batchSize = 6;

  final OrtSession session;

  TextClassifier(this.session);

  Future<List<ClassificationOutput>> classifyAndRotate(
    List<img.Image> images,
  ) async {
    final results = <ClassificationOutput>[];

    for (int i = 0; i < images.length; i += batchSize) {
      final batchEnd = (i + batchSize).clamp(0, images.length);
      final batch = images.sublist(i, batchEnd);

      final rotationFlags = await classifyBatch(batch);

      for (int j = 0; j < batch.length; j++) {
        final image = batch[j];
        final shouldRotate = rotationFlags[j];

        results.add(
          shouldRotate
              ? ClassificationOutput(rotateImage180(image), true)
              : ClassificationOutput(image, false),
        );
      }
    }

    return results;
  }

  Future<List<bool>> classifyBatch(List<img.Image> batchImages) async {
    if (batchImages.isEmpty) return [];

    final batchSz = batchImages.length;
    final inputArray = Float32List(batchSz * 3 * imgHeight * imgWidth);

    for (int index = 0; index < batchImages.length; index++) {
      preprocessImage(batchImages[index], inputArray, index);
    }

    final shape = [batchSz, 3, imgHeight, imgWidth];
    final inputTensor = await OrtValue.fromList(inputArray, shape);

    final inputs = {session.inputNames.first: inputTensor};
    final outputs = await session.run(inputs);
    final output = outputs.values.first;

    final results = await decodeOutput(output, batchSz);

    await inputTensor.dispose();
    await output.dispose();

    return results;
  }

  void preprocessImage(
    img.Image bitmap,
    Float32List outputArray,
    int batchIndex,
  ) {
    final aspectRatio = bitmap.width / bitmap.height;
    final resizedWidth = (imgHeight * aspectRatio).ceil().clamp(1, imgWidth);

    final resized = img.copyResize(
      bitmap,
      width: resizedWidth,
      height: imgHeight,
      interpolation: img.Interpolation.linear,
    );

    final baseOffset = batchIndex * 3 * imgHeight * imgWidth;
    final channelStride = imgHeight * imgWidth;

    for (int y = 0; y < imgHeight; y++) {
      final rowOffset = y * imgWidth;

      for (int x = 0; x < imgWidth; x++) {
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
  }

  Future<List<bool>> decodeOutput(OrtValue output, int batchSz) async {
    final outputData = await output.asFlattenedList();
    final results = <bool>[];

    for (int b = 0; b < batchSz; b++) {
      final baseOffset = b * 2;
      final prob0 = (outputData[baseOffset] as num).toDouble();
      final prob180 = (outputData[baseOffset + 1] as num).toDouble();

      final shouldRotate = prob180 > prob0 && prob180 > clsThresh;
      results.add(shouldRotate);
    }

    return results;
  }

  img.Image rotateImage180(img.Image bitmap) {
    return img.copyRotate(bitmap, angle: 180);
  }
}
