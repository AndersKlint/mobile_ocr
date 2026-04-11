part of 'text_detector_widget.dart';

/// Controller that surfaces imperative actions for [TextDetectorWidget].
class TextDetectorController extends ChangeNotifier {
  _TextDetectorWidgetState? _state;

  void _attach(_TextDetectorWidgetState state) {
    if (identical(_state, state)) {
      return;
    }
    _state = state;
    scheduleMicrotask(() {
      notifyListeners();
    });
  }

  void _detach(_TextDetectorWidgetState state) {
    if (identical(_state, state)) {
      _state = null;
      notifyListeners();
    }
  }

  void _notifyStateChanged() {
    notifyListeners();
  }

  /// Whether text detection is currently running.
  bool get isProcessing => _state?._isProcessing ?? false;

  /// Indicates if there is text that can be selected.
  bool get hasSelectableText => _state?._hasSelectableText ?? false;

  /// Programmatically select all recognized text.
  bool selectAllText() {
    final state = _state;
    if (state == null) {
      return false;
    }
    return state._selectAllRecognizedText();
  }
}
