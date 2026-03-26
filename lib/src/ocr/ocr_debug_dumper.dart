import 'dart:convert';
import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:image/image.dart' as img;

class OcrDebugSession {
  final Directory rootDirectory;

  OcrDebugSession._(this.rootDirectory);

  String get path => rootDirectory.path;

  static Future<OcrDebugSession?> create(String? baseDirPath) async {
    if (baseDirPath == null || baseDirPath.isEmpty) {
      return null;
    }

    try {
      final baseDirectory = Directory(baseDirPath);
      await baseDirectory.create(recursive: true);

      final latestDirectory = Directory(
        '${baseDirectory.path}${Platform.pathSeparator}latest',
      );
      if (await latestDirectory.exists()) {
        await latestDirectory.delete(recursive: true);
      }
      await latestDirectory.create(recursive: true);

      debugPrint('Mobile OCR debug dump path: ${latestDirectory.path}');
      return OcrDebugSession._(latestDirectory);
    } catch (error) {
      debugPrint('Mobile OCR: failed to create debug dump directory: $error');
      return null;
    }
  }

  Future<void> saveImage(String relativePath, img.Image image) async {
    try {
      final file = await _ensureFile(relativePath);
      await file.writeAsBytes(img.encodePng(image), flush: true);
    } catch (error) {
      debugPrint(
        'Mobile OCR: failed to save debug image $relativePath: $error',
      );
    }
  }

  Future<void> writeJson(String relativePath, Object? value) async {
    try {
      final file = await _ensureFile(relativePath);
      final encoder = const JsonEncoder.withIndent('  ');
      await file.writeAsString(encoder.convert(value), flush: true);
    } catch (error) {
      debugPrint('Mobile OCR: failed to save debug JSON $relativePath: $error');
    }
  }

  Future<void> writeText(String relativePath, String value) async {
    try {
      final file = await _ensureFile(relativePath);
      await file.writeAsString(value, flush: true);
    } catch (error) {
      debugPrint('Mobile OCR: failed to save debug text $relativePath: $error');
    }
  }

  Future<File> _ensureFile(String relativePath) async {
    final normalizedPath = relativePath.replaceAll(
      '\\',
      Platform.pathSeparator,
    );
    final file = File(
      '${rootDirectory.path}${Platform.pathSeparator}$normalizedPath',
    );
    await file.parent.create(recursive: true);
    return file;
  }
}

img.Image buildModelInputPreview({
  required img.Image resizedImage,
  required int targetWidth,
  required int targetHeight,
  int padValue = 127,
}) {
  final preview = img.Image(width: targetWidth, height: targetHeight);
  img.fill(preview, color: img.ColorRgb8(padValue, padValue, padValue));

  final width = resizedImage.width < targetWidth
      ? resizedImage.width
      : targetWidth;
  final height = resizedImage.height < targetHeight
      ? resizedImage.height
      : targetHeight;

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      preview.setPixel(x, y, resizedImage.getPixel(x, y));
    }
  }

  return preview;
}
