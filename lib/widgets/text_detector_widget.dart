import 'dart:async';
import 'dart:io';

import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'package:mobile_ocr/mobile_ocr_plugin.dart';
import 'package:mobile_ocr/models/text_block.dart';
import 'package:mobile_ocr/src/display_image_helper.dart';
import 'package:mobile_ocr/widgets/text_overlay_widget.dart';

part 'text_detector_widget_controller.dart';
part 'text_detector_widget_image_layer.dart';
part 'text_detector_widget_status_widgets.dart';
part 'text_detector_widget_strings.dart';

const Color _selectionPrimaryColor = Color(0xFF1DB954);
const double _selectionHighlightOpacity = 0.28;

/// A complete text detection widget that displays an image and allows
/// users to select and copy detected text.
class TextDetectorWidget extends StatefulWidget {
  /// The path to the image file to detect text from
  final String imagePath;

  /// Callback when text is copied
  final Function(String)? onTextCopied;

  /// Callback when text blocks are selected
  final Function(List<TextBlock>)? onTextBlocksSelected;

  /// Whether to auto-detect text on load
  final bool autoDetect;

  /// Background color
  final Color backgroundColor;

  /// Whether to show boundaries for unselected text
  final bool showUnselectedBoundaries;

  /// Whether to show the inline selection preview banner.
  final bool enableSelectionPreview;

  /// Enable debug utilities like the detected-text dialog.
  final bool debugMode;

  /// Strings used for user-facing text in the widget.
  final TextDetectorStrings strings;

  /// Optional directory where the OCR pipeline writes debug images of the
  /// actual inputs sent to the detector/classifier/recognizer models.
  final String? debugDumpDir;

  /// Controller for imperative text selection actions.
  final TextDetectorController? controller;

  const TextDetectorWidget({
    super.key,
    required this.imagePath,
    this.onTextCopied,
    this.onTextBlocksSelected,
    this.autoDetect = true,
    this.backgroundColor = Colors.transparent,
    this.showUnselectedBoundaries = true,
    this.enableSelectionPreview = false,
    this.debugMode = false,
    this.strings = const TextDetectorStrings(),
    this.debugDumpDir,
    this.controller,
  });

  @override
  State<TextDetectorWidget> createState() => _TextDetectorWidgetState();
}

class _TextDetectorWidgetState extends State<TextDetectorWidget> {
  final MobileOcr _ocr = MobileOcr();
  final TextOverlayController _textOverlayController = TextOverlayController();
  List<TextBlock>? _detectedTextBlocks;
  bool _isProcessing = false;
  File? _imageFile;
  String? _resolvedImagePath;
  Future<void>? _imagePreparation;
  bool _modelsReady = false;
  Future<void>? _modelPreparation;
  String? _errorMessage;
  Timer? _editorHintTimer;
  bool _showEditorHint = false;
  bool _isNetworkError = false;
  bool get _hasSelectableText =>
      _detectedTextBlocks != null && _detectedTextBlocks!.isNotEmpty;

  @override
  void initState() {
    super.initState();
    widget.controller?._attach(this);
    // Set initial processing state if auto-detecting
    if (widget.autoDetect) {
      _isProcessing = true;
    }
    // Schedule file initialization after first frame to ensure immediate rendering
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (!mounted) return;
      unawaited(_initializeFile());
    });
  }

  @override
  void dispose() {
    _editorHintTimer?.cancel();
    widget.controller?._detach(this);
    super.dispose();
  }

  Future<void> _initializeFile() async {
    final requestedPath = widget.imagePath;
    _editorHintTimer?.cancel();

    final preparation = _prepareDisplayImage(requestedPath);
    _imagePreparation = preparation;
    await preparation;
    if (_imagePreparation == preparation) {
      _imagePreparation = null;
    }
  }

  Future<void> _prepareDisplayImage(String requestedPath) async {
    setState(() {
      _imageFile = null;
      _resolvedImagePath = null;
      _showEditorHint = false;
      _errorMessage = null;
    });

    try {
      final resolvedPath = await DisplayImageHelper.ensureDisplayablePath(
        requestedPath,
      );
      if (!mounted || widget.imagePath != requestedPath) {
        return;
      }
      final file = File(resolvedPath);
      if (!file.existsSync()) {
        throw Exception('Image file not found after normalization');
      }

      setState(() {
        _imageFile = file;
        _resolvedImagePath = resolvedPath;
      });

      _precacheCurrentImage();

      if (widget.autoDetect) {
        unawaited(_detectText());
      }
    } catch (error) {
      debugPrint('Failed to prepare image $requestedPath: $error');
      if (!mounted || widget.imagePath != requestedPath) {
        return;
      }
      setState(() {
        _imageFile = null;
        _resolvedImagePath = null;
        _errorMessage = widget.strings.imageDecodeFailedError;
        _isProcessing = false;
      });
    }
  }

  @override
  void didUpdateWidget(covariant TextDetectorWidget oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.controller != widget.controller) {
      oldWidget.controller?._detach(this);
      widget.controller?._attach(this);
    }
    if (oldWidget.imagePath != widget.imagePath) {
      setState(() {
        _isProcessing = widget.autoDetect;
        _detectedTextBlocks = null;
        _imageFile = null;
        _errorMessage = null;
        _isNetworkError = false;
      });
      _notifyController();
      unawaited(_initializeFile());
    }
  }

  Future<void> _ensureModelsReady() async {
    if (_modelsReady) return;

    _modelPreparation ??= _ocr
        .prepareModels()
        .then((status) {
          if (!status.isReady) {
            _applyModelPreparationFailure(status.errorMessage);
            return;
          }
          _modelsReady = status.isReady;
        })
        .catchError((error, _) {
          _applyModelPreparationFailure(error.toString());
          debugPrint('Model preparation error: $error');
        })
        .whenComplete(() {
          _modelPreparation = null;
        });

    await _modelPreparation;
  }

  void _applyModelPreparationFailure(String? errorMessage) {
    final errorStr = errorMessage?.toLowerCase() ?? '';
    _isNetworkError =
        errorStr.contains('network') ||
        errorStr.contains('connection') ||
        errorStr.contains('timeout') ||
        errorStr.contains('failed to download') ||
        errorStr.contains('http');

    _errorMessage = _isNetworkError
        ? widget.strings.modelsNetworkRequiredError
        : widget.strings.modelsPrepareFailed;
  }

  Future<void> _detectText() async {
    final String requestedPath = widget.imagePath;
    String? imagePath = _resolvedImagePath;
    if (imagePath == null) {
      final pendingPreparation = _imagePreparation;
      if (pendingPreparation != null) {
        await pendingPreparation;
        if (!mounted || widget.imagePath != requestedPath) {
          return;
        }
        imagePath = _resolvedImagePath;
      }
    }
    if (imagePath == null) {
      setState(() {
        _errorMessage = widget.strings.imageDecodeFailedError;
        _isProcessing = false;
      });
      _notifyController();
      return;
    }

    // Don't set processing true here if already processing
    if (!_isProcessing) {
      setState(() {
        _isProcessing = true;
        _detectedTextBlocks = null;
        _errorMessage = null;
        _isNetworkError = false;
      });
      _notifyController();
    }

    try {
      await _ensureModelsReady();
      if (_errorMessage != null) {
        throw Exception(_errorMessage);
      }

      final blocks = await _ocr.detectText(
        imagePath: imagePath,
        debugDumpDir: widget.debugDumpDir,
        enhanceRecognitionCrops: true,
      );

      if (mounted && widget.imagePath == requestedPath) {
        setState(() {
          _detectedTextBlocks = blocks;
          _errorMessage = null;
        });
        _notifyController();
        _handleEditorHint(blocks);
      }
    } catch (e) {
      debugPrint('Error detecting text: $e');
      if (mounted && widget.imagePath == requestedPath) {
        setState(() {
          // Show user-friendly message based on error type
          final errorStr = e.toString().toLowerCase();
          if (errorStr.contains('image') &&
              errorStr.contains('not') &&
              errorStr.contains('exist')) {
            _errorMessage = widget.strings.imageNotFoundError;
          } else if (errorStr.contains('failed to decode')) {
            _errorMessage = widget.strings.imageDecodeFailedError;
          } else {
            _errorMessage = widget.strings.genericDetectError;
          }
        });
        _notifyController();
      }
    } finally {
      if (mounted && widget.imagePath == requestedPath) {
        setState(() {
          _isProcessing = false;
        });
        _notifyController();
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Stack(
      fit: StackFit.expand,
      children: [
        _buildImageLayer(),
        if (_isProcessing && _detectedTextBlocks == null)
          _ProcessingOverlay(message: widget.strings.processingOverlayMessage),
        if (_showEditorHint &&
            _detectedTextBlocks != null &&
            _detectedTextBlocks!.isNotEmpty)
          _EditorHintOverlay(message: widget.strings.selectionHint),
        if (_errorMessage != null)
          Positioned(
            bottom: 32,
            left: 16,
            right: 16,
            child: _isNetworkError
                ? _NetworkErrorBanner(
                    message: _errorMessage!,
                    retryLabel: widget.strings.retryButtonLabel,
                    onRetry: _retryAfterNetworkError,
                  )
                : _ErrorBanner(message: _errorMessage!),
          ),
        if (_detectedTextBlocks != null &&
            _detectedTextBlocks!.isEmpty &&
            _errorMessage == null)
          Positioned(
            top: 100,
            left: 0,
            right: 0,
            child: Center(
              child: _NoTextMessage(message: widget.strings.noTextDetected),
            ),
          ),
      ],
    );
  }

  void _handleEditorHint(List<TextBlock> blocks) {
    _editorHintTimer?.cancel();
    if (!mounted) {
      return;
    }

    if (blocks.isEmpty) {
      if (_showEditorHint) {
        setState(() {
          _showEditorHint = false;
        });
      }
      return;
    }

    setState(() {
      _showEditorHint = true;
    });

    _editorHintTimer = Timer(const Duration(seconds: 3), () {
      if (!mounted) {
        return;
      }
      setState(() {
        _showEditorHint = false;
      });
    });
  }

  void _dismissEditorHint() {
    if (!_showEditorHint) {
      return;
    }
    _editorHintTimer?.cancel();
    if (!mounted) {
      return;
    }
    setState(() {
      _showEditorHint = false;
    });
  }

  void _retryAfterNetworkError() {
    setState(() {
      _errorMessage = null;
      _isNetworkError = false;
      _modelsReady = false;
    });
    _detectText();
  }

  /// Manually trigger text detection
  Future<void> detectText() {
    return _detectText();
  }

  /// Get the currently detected text blocks
  List<TextBlock>? get detectedTextBlocks => _detectedTextBlocks;

  /// Check if text detection is currently processing
  bool get isProcessing => _isProcessing;

  bool _selectAllRecognizedText() {
    if (!_hasSelectableText) {
      return false;
    }
    return _textOverlayController.selectAllText();
  }

  void _notifyController() {
    widget.controller?._notifyStateChanged();
  }

  void _precacheCurrentImage() {
    final imageFile = _imageFile;
    if (imageFile == null) {
      return;
    }
    precacheImage(FileImage(imageFile), context);
  }
}
