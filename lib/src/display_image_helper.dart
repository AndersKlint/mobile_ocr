import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';

class DisplayImageHelper {
  DisplayImageHelper._();

  static const MethodChannel _channel = MethodChannel('mobile_ocr');
  static final Map<String, String> _cache = <String, String>{};
  static final Map<String, Future<String>> _inFlight =
      <String, Future<String>>{};

  static bool get _needsPlatformConversion =>
      !Platform.isLinux && !Platform.isWindows && !Platform.isMacOS;

  static Future<String> ensureDisplayablePath(String imagePath) {
    final cached = _cache[imagePath];
    if (cached != null) {
      return SynchronousFuture<String>(cached);
    }

    if (!_needsPlatformConversion) {
      return SynchronousFuture<String>(imagePath);
    }

    final inflight = _inFlight[imagePath];
    if (inflight != null) {
      return inflight;
    }

    final future = _invokePlatform(imagePath);
    _inFlight[imagePath] = future;
    return future;
  }

  static Future<String> _invokePlatform(String imagePath) async {
    try {
      final resolved = await _channel.invokeMethod<String>(
        'ensureImageIsDisplayable',
        {'imagePath': imagePath},
      );
      final result = (resolved == null || resolved.isEmpty)
          ? imagePath
          : resolved;
      _cache[imagePath] = result;
      return result;
    } catch (error, stack) {
      debugPrint('Failed to normalize image $imagePath: $error');
      debugPrintStack(stackTrace: stack);
      return imagePath;
    } finally {
      _inFlight.remove(imagePath);
    }
  }
}
