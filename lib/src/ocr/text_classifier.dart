import 'dart:typed_data';
import 'package:onnxruntime_v2/onnxruntime_v2.dart';
import 'package:image/image.dart' as img;
import 'fast_image_loader.dart';
import 'fast_tensor_reader.dart';
import 'ocr_debug_dumper.dart';

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
  static const List<double> _mean = [0.5, 0.5, 0.5];
  static const List<double> _std = [0.5, 0.5, 0.5];

  final OrtSession session;
  final OcrDebugSession? debugSession;

  TextClassifier(this.session, {this.debugSession});

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
    inputArray.fillRange(0, inputArray.length, 0);

    for (int index = 0; index < batchImages.length; index++) {
      await preprocessImage(batchImages[index], inputArray, index);
    }

    final shape = [batchSz, 3, imgHeight, imgWidth];
    final inputTensor = OrtValueTensor.createTensorWithDataList(
      inputArray,
      shape,
    );

    final inputs = {session.inputNames.first: inputTensor};
    final runOptions = OrtRunOptions();
    final outputs = session.run(runOptions, inputs);
    runOptions.release();
    final output = outputs[0];

    List<bool> results = [];
    if (output != null) {
      results = decodeOutput(output, batchSz);
      output.release();
    }

    inputTensor.release();

    return results;
  }

  Future<void> preprocessImage(
    img.Image bitmap,
    Float32List outputArray,
    int batchIndex,
  ) async {
    final aspectRatio = bitmap.width / bitmap.height;
    final resizedWidth = (imgHeight * aspectRatio).ceil().clamp(1, imgWidth);
    final baseOffset = batchIndex * 3 * imgHeight * imgWidth;

    final wroteTensor = await FastImageLoader.resizeToNormalizedBatchBuffer(
      bitmap,
      resizedWidth: resizedWidth,
      targetWidth: imgWidth,
      targetHeight: imgHeight,
      buffer: outputArray,
      bufferOffset: baseOffset,
      mean: _mean,
      std: _std,
      bgrOrder: true,
    );
    if (!wroteTensor) {
      return;
    }

    if (debugSession != null) {
      final resized = img.copyResize(
        bitmap,
        width: resizedWidth,
        height: imgHeight,
        interpolation: img.Interpolation.linear,
      );
      final modelInputPreview = buildModelInputPreview(
        resizedImage: resized,
        targetWidth: imgWidth,
        targetHeight: imgHeight,
      );
      await debugSession!.saveImage(
        '02_classifier/${batchIndex.toString().padLeft(3, '0')}_crop.png',
        bitmap,
      );
      await debugSession!.saveImage(
        '02_classifier/${batchIndex.toString().padLeft(3, '0')}_model_input.png',
        modelInputPreview,
      );
      await debugSession!.writeJson(
        '02_classifier/${batchIndex.toString().padLeft(3, '0')}_meta.json',
        {
          'originalWidth': bitmap.width,
          'originalHeight': bitmap.height,
          'resizedWidth': resizedWidth,
          'targetWidth': imgWidth,
          'targetHeight': imgHeight,
          'aspectRatio': aspectRatio,
        },
      );
    }
  }

  List<bool> decodeOutput(OrtValue output, int batchSz) {
    final flatData = FastTensorReader.asFloat32List(output);
    if (flatData == null || flatData.isEmpty) return [];

    final results = <bool>[];

    for (int b = 0; b < batchSz; b++) {
      final baseOffset = b * 2;
      if (baseOffset + 1 >= flatData.length) break;

      final prob0 = flatData[baseOffset];
      final prob180 = flatData[baseOffset + 1];

      final shouldRotate = prob180 > prob0 && prob180 > clsThresh;
      results.add(shouldRotate);
    }

    return results;
  }

  img.Image rotateImage180(img.Image bitmap) {
    return img.copyRotate(bitmap, angle: 180);
  }
}
