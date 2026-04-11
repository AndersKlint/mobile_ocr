part of 'text_detector_widget.dart';

extension _TextDetectorImageLayer on _TextDetectorWidgetState {
  Widget _buildImageLayer() {
    final imageFile = _imageFile;
    final textBlocks = _detectedTextBlocks;
    if (imageFile == null || textBlocks == null) {
      return const SizedBox.shrink();
    }

    final TextSelectionThemeData baseSelectionTheme = TextSelectionTheme.of(
      context,
    );
    final TextSelectionThemeData overlaySelectionTheme = baseSelectionTheme
        .copyWith(
          selectionColor: _selectionPrimaryColor.withValues(
            alpha: _selectionHighlightOpacity,
          ),
          selectionHandleColor: _selectionPrimaryColor,
        );

    return Container(
      color: widget.backgroundColor,
      child: TextSelectionTheme(
        data: overlaySelectionTheme,
        child: TextOverlayWidget(
          imageFile: imageFile,
          textBlocks: textBlocks,
          onTextBlocksSelected: widget.onTextBlocksSelected,
          onTextCopied: widget.onTextCopied,
          onSelectionStart: _dismissEditorHint,
          showUnselectedBoundaries: widget.showUnselectedBoundaries,
          enableSelectionPreview: widget.enableSelectionPreview,
          debugMode: widget.debugMode,
          controller: _textOverlayController,
        ),
      ),
    );
  }
}
