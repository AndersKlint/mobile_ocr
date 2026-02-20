import 'dart:io';
import 'package:plugin_platform_interface/plugin_platform_interface.dart';

import 'mobile_ocr_plugin_method_channel.dart';
import 'mobile_ocr_plugin_dart.dart';

abstract class MobileOcrPlatform extends PlatformInterface {
  /// Constructs a MobileOcrPlatform.
  MobileOcrPlatform() : super(token: _token);

  static final Object _token = Object();

  static MobileOcrPlatform _instance = _createDefaultInstance();

  static MobileOcrPlatform _createDefaultInstance() {
    if (Platform.isLinux ||
        Platform.isMacOS ||
        Platform.isWindows ||
        Platform.isIOS) {
      return DartMobileOcr();
    }
    return MethodChannelMobileOcr();
  }

  /// The default instance of [MobileOcrPlatform] to use.
  ///
  /// Defaults to [MethodChannelMobileOcr] on Android,
  /// [DartMobileOcr] on iOS, Linux, macOS, and Windows.
  static MobileOcrPlatform get instance => _instance;

  /// Platform-specific implementations should set this with their own
  /// platform-specific class that extends [MobileOcrPlatform] when
  /// they register themselves.
  static set instance(MobileOcrPlatform instance) {
    PlatformInterface.verifyToken(instance, _token);
    _instance = instance;
  }

  Future<String?> getPlatformVersion() {
    throw UnimplementedError('platformVersion() has not been implemented.');
  }

  Future<List<Map<dynamic, dynamic>>> detectText({
    required String imagePath,
    bool includeAllConfidenceScores = false,
  }) {
    throw UnimplementedError('detectText() has not been implemented.');
  }

  Future<bool> hasText({required String imagePath}) {
    throw UnimplementedError('hasText() has not been implemented.');
  }

  Future<Map<dynamic, dynamic>> prepareModels() {
    throw UnimplementedError('prepareModels() has not been implemented.');
  }
}
