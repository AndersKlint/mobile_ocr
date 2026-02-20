import 'dart:async';
import 'dart:typed_data';
import 'dart:ui' as ui;
import 'package:image/image.dart' as img;

class FastImageLoader {
  static Future<img.Image?> loadFromBytes(Uint8List bytes) async {
    final codec = await ui.instantiateImageCodec(bytes);
    final frame = await codec.getNextFrame();
    final uiImage = frame.image;

    final byteData = await uiImage.toByteData(
      format: ui.ImageByteFormat.rawRgba,
    );
    if (byteData == null) {
      codec.dispose();
      return null;
    }

    final rgba = byteData.buffer.asUint8List();
    final image = img.Image(width: uiImage.width, height: uiImage.height);
    _fillImageFromRgba(image, rgba);

    codec.dispose();
    return image;
  }

  static Future<(Uint8List, int, int)?> loadAsRgba(
    Uint8List bytes, {
    int? targetWidth,
    int? targetHeight,
  }) async {
    final codec = await ui.instantiateImageCodec(
      bytes,
      targetWidth: targetWidth,
      targetHeight: targetHeight,
      allowUpscaling: true,
    );
    final frame = await codec.getNextFrame();
    final uiImage = frame.image;

    final byteData = await uiImage.toByteData(
      format: ui.ImageByteFormat.rawRgba,
    );
    if (byteData == null) {
      codec.dispose();
      return null;
    }

    final rgba = byteData.buffer.asUint8List().toList();
    codec.dispose();
    return (Uint8List.fromList(rgba), uiImage.width, uiImage.height);
  }

  static Future<(Float32List, int, int)?> loadBytesAsTensorDirect(
    Uint8List bytes, {
    required int maxSide,
    required int multipleOf,
    required List<double> mean,
    required List<double> std,
    bool bgrOrder = true,
  }) async {
    final codec = await ui.instantiateImageCodec(bytes);
    final frame = await codec.getNextFrame();
    final originalImage = frame.image;

    final originalWidth = originalImage.width;
    final originalHeight = originalImage.height;

    final maxSideOrig = originalWidth > originalHeight
        ? originalWidth
        : originalHeight;
    final ratio = maxSideOrig > maxSide ? maxSide / maxSideOrig : 1.0;

    var resizedWidth = (originalWidth * ratio).round().clamp(1, 10000);
    var resizedHeight = (originalHeight * ratio).round().clamp(1, 10000);

    resizedWidth =
        (((resizedWidth + multipleOf - 1) ~/ multipleOf) * multipleOf).clamp(
          multipleOf,
          10000,
        );
    resizedHeight =
        (((resizedHeight + multipleOf - 1) ~/ multipleOf) * multipleOf).clamp(
          multipleOf,
          10000,
        );

    codec.dispose();

    final resizeCodec = await ui.instantiateImageCodec(
      bytes,
      targetWidth: resizedWidth,
      targetHeight: resizedHeight,
      allowUpscaling: true,
    );
    final resizeFrame = await resizeCodec.getNextFrame();
    final resizedImage = resizeFrame.image;

    final byteData = await resizedImage.toByteData(
      format: ui.ImageByteFormat.rawRgba,
    );
    if (byteData == null) {
      resizeCodec.dispose();
      return null;
    }

    final rgba = byteData.buffer.asUint8List();
    final inputArray = _rgbaToTensor(
      rgba,
      resizedWidth,
      resizedHeight,
      mean,
      std,
      bgrOrder,
    );

    resizeCodec.dispose();
    return (inputArray, resizedWidth, resizedHeight);
  }

  static Future<img.Image?> cropAndResize(
    img.Image source, {
    required int srcX,
    required int srcY,
    required int srcWidth,
    required int srcHeight,
    required int targetWidth,
    required int targetHeight,
  }) async {
    final rgba = Uint8List(source.width * source.height * 4);
    _imageToRgba(source, rgba);

    final completer = Completer<ui.Image>();
    ui.decodeImageFromPixels(
      rgba,
      source.width,
      source.height,
      ui.PixelFormat.rgba8888,
      completer.complete,
    );

    final uiSourceImage = await completer.future;

    final recorder = ui.PictureRecorder();
    final canvas = ui.Canvas(recorder);
    final paint = ui.Paint()..filterQuality = ui.FilterQuality.low;

    final srcRect = ui.Rect.fromLTWH(
      srcX.toDouble(),
      srcY.toDouble(),
      srcWidth.toDouble(),
      srcHeight.toDouble(),
    );
    final dstRect = ui.Rect.fromLTWH(
      0,
      0,
      targetWidth.toDouble(),
      targetHeight.toDouble(),
    );

    canvas.drawImageRect(uiSourceImage, srcRect, dstRect, paint);

    final picture = recorder.endRecording();
    final resizedUiImage = await picture.toImage(targetWidth, targetHeight);

    final byteData = await resizedUiImage.toByteData(
      format: ui.ImageByteFormat.rawRgba,
    );
    if (byteData == null) {
      return null;
    }

    final resizedRgba = byteData.buffer.asUint8List();
    final result = img.Image(width: targetWidth, height: targetHeight);
    _fillImageFromRgba(result, resizedRgba);

    return result;
  }

  static Future<Float32List?> cropResizeToTensor(
    img.Image source, {
    required int srcX,
    required int srcY,
    required int srcWidth,
    required int srcHeight,
    required int targetWidth,
    required int targetHeight,
    required List<double> mean,
    required List<double> std,
    bool bgrOrder = false,
  }) async {
    final rgba = Uint8List(source.width * source.height * 4);
    _imageToRgba(source, rgba);

    final completer = Completer<ui.Image>();
    ui.decodeImageFromPixels(
      rgba,
      source.width,
      source.height,
      ui.PixelFormat.rgba8888,
      completer.complete,
    );

    final uiSourceImage = await completer.future;

    final recorder = ui.PictureRecorder();
    final canvas = ui.Canvas(recorder);
    final paint = ui.Paint()..filterQuality = ui.FilterQuality.low;

    final srcRect = ui.Rect.fromLTWH(
      srcX.toDouble(),
      srcY.toDouble(),
      srcWidth.toDouble(),
      srcHeight.toDouble(),
    );
    final dstRect = ui.Rect.fromLTWH(
      0,
      0,
      targetWidth.toDouble(),
      targetHeight.toDouble(),
    );

    canvas.drawImageRect(uiSourceImage, srcRect, dstRect, paint);

    final picture = recorder.endRecording();
    final resizedUiImage = await picture.toImage(targetWidth, targetHeight);

    final byteData = await resizedUiImage.toByteData(
      format: ui.ImageByteFormat.rawRgba,
    );
    if (byteData == null) {
      return null;
    }

    final resizedRgba = byteData.buffer.asUint8List();
    return _rgbaToTensor(
      resizedRgba,
      targetWidth,
      targetHeight,
      mean,
      std,
      bgrOrder,
    );
  }

  static Future<Float32List?> imageToTensor(
    img.Image source, {
    required int targetWidth,
    required int targetHeight,
    required List<double> mean,
    required List<double> std,
    bool bgrOrder = false,
  }) async {
    return cropResizeToTensor(
      source,
      srcX: 0,
      srcY: 0,
      srcWidth: source.width,
      srcHeight: source.height,
      targetWidth: targetWidth,
      targetHeight: targetHeight,
      mean: mean,
      std: std,
      bgrOrder: bgrOrder,
    );
  }

  static void _imageToRgba(img.Image source, Uint8List rgba) {
    final width = source.width;
    final height = source.height;
    for (int y = 0; y < height; y++) {
      final rowOffset = y * width;
      for (int x = 0; x < width; x++) {
        final pixel = source.getPixel(x, y);
        final idx = (rowOffset + x) * 4;
        rgba[idx] = pixel.r.toInt();
        rgba[idx + 1] = pixel.g.toInt();
        rgba[idx + 2] = pixel.b.toInt();
        rgba[idx + 3] = pixel.a.toInt();
      }
    }
  }

  static void _fillImageFromRgba(img.Image image, Uint8List rgba) {
    final width = image.width;
    final height = image.height;
    for (int y = 0; y < height; y++) {
      final rowOffset = y * width;
      for (int x = 0; x < width; x++) {
        final srcIdx = (rowOffset + x) * 4;
        image.setPixelRgba(
          x,
          y,
          rgba[srcIdx],
          rgba[srcIdx + 1],
          rgba[srcIdx + 2],
          rgba[srcIdx + 3],
        );
      }
    }
  }

  static Float32List _rgbaToTensor(
    Uint8List rgba,
    int width,
    int height,
    List<double> mean,
    List<double> std,
    bool bgrOrder,
  ) {
    final inputArray = Float32List(3 * height * width);
    _rgbaToTensorBuffer(
      rgba,
      width,
      height,
      mean,
      std,
      bgrOrder,
      inputArray,
      0,
    );
    return inputArray;
  }

  static void _rgbaToTensorBuffer(
    Uint8List rgba,
    int width,
    int height,
    List<double> mean,
    List<double> std,
    bool bgrOrder,
    Float32List buffer,
    int bufferOffset,
  ) {
    final channelStride = height * width;
    const scale = 1.0 / 255.0;

    for (int y = 0; y < height; y++) {
      final rowOffset = y * width;
      for (int x = 0; x < width; x++) {
        final srcIdx = (rowOffset + x) * 4;
        final r = rgba[srcIdx].toDouble() * scale;
        final g = rgba[srcIdx + 1].toDouble() * scale;
        final b = rgba[srcIdx + 2].toDouble() * scale;

        final pixelIndex = bufferOffset + rowOffset + x;

        if (bgrOrder) {
          buffer[pixelIndex] = (b - mean[0]) / std[0];
          buffer[pixelIndex + channelStride] = (g - mean[1]) / std[1];
          buffer[pixelIndex + 2 * channelStride] = (r - mean[2]) / std[2];
        } else {
          buffer[pixelIndex] = (r - mean[0]) / std[0];
          buffer[pixelIndex + channelStride] = (g - mean[1]) / std[1];
          buffer[pixelIndex + 2 * channelStride] = (b - mean[2]) / std[2];
        }
      }
    }
  }

  static Future<int?> resizeToBuffer(
    img.Image source, {
    required int targetWidth,
    required int targetHeight,
    required Float32List buffer,
    required int bufferOffset,
    required List<double> mean,
    required List<double> std,
    bool bgrOrder = false,
  }) async {
    final rgba = Uint8List(source.width * source.height * 4);
    _imageToRgba(source, rgba);

    final completer = Completer<ui.Image>();
    ui.decodeImageFromPixels(
      rgba,
      source.width,
      source.height,
      ui.PixelFormat.rgba8888,
      completer.complete,
    );

    final uiSourceImage = await completer.future;

    final recorder = ui.PictureRecorder();
    final canvas = ui.Canvas(recorder);
    final paint = ui.Paint()..filterQuality = ui.FilterQuality.low;

    final srcRect = ui.Rect.fromLTWH(
      0,
      0,
      source.width.toDouble(),
      source.height.toDouble(),
    );
    final dstRect = ui.Rect.fromLTWH(
      0,
      0,
      targetWidth.toDouble(),
      targetHeight.toDouble(),
    );

    canvas.drawImageRect(uiSourceImage, srcRect, dstRect, paint);

    final picture = recorder.endRecording();
    final resizedUiImage = await picture.toImage(targetWidth, targetHeight);

    final byteData = await resizedUiImage.toByteData(
      format: ui.ImageByteFormat.rawRgba,
    );
    if (byteData == null) {
      return null;
    }

    final resizedRgba = byteData.buffer.asUint8List();
    _rgbaToTensorBuffer(
      resizedRgba,
      targetWidth,
      targetHeight,
      mean,
      std,
      bgrOrder,
      buffer,
      bufferOffset,
    );

    return source.width;
  }
}
