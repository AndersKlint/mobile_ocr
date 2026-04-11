part of 'text_detector_widget.dart';

/// Collection of user-facing strings used by [TextDetectorWidget].
class TextDetectorStrings {
  final String processingOverlayMessage;
  final String selectionHint;
  final String noTextDetected;
  final String retryButtonLabel;
  final String modelsNetworkRequiredError;
  final String modelsPrepareFailed;
  final String imageNotFoundError;
  final String imageDecodeFailedError;
  final String genericDetectError;

  const TextDetectorStrings({
    this.processingOverlayMessage = 'Detecting text...',
    this.selectionHint = 'Swipe or double tap to select just what you need',
    this.noTextDetected = 'No text detected',
    this.retryButtonLabel = 'Retry',
    this.modelsNetworkRequiredError =
        'Network connection required to download OCR models on first use',
    this.modelsPrepareFailed = 'Could not prepare OCR models',
    this.imageNotFoundError = 'Image file not found',
    this.imageDecodeFailedError = 'Could not read image file',
    this.genericDetectError = 'Could not detect text in image',
  });
}
