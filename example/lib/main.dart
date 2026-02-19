import 'dart:io';

import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image_picker/image_picker.dart';
import 'package:mobile_ocr/mobile_ocr.dart';
import 'package:path_provider/path_provider.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  if (Platform.isLinux || Platform.isWindows || Platform.isMacOS) {
    await _prepareModelsForDesktop();
  }

  runApp(const MyApp());
}

Future<void> _prepareModelsForDesktop() async {
  final appDir = await getApplicationSupportDirectory();
  final modelsDir = Directory('${appDir.path}/assets/mobile_ocr');

  if (!await modelsDir.exists()) {
    await modelsDir.create(recursive: true);
  }

  const modelFiles = ['det.onnx', 'rec.onnx', 'cls.onnx', 'ppocrv5_dict.txt'];

  for (final modelFile in modelFiles) {
    final targetFile = File('${modelsDir.path}/$modelFile');
    if (!await targetFile.exists()) {
      final data = await rootBundle.load('assets/mobile_ocr/$modelFile');
      await targetFile.writeAsBytes(
        data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes),
        flush: true,
      );
    }
  }
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Mobile OCR Demo',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.indigo),
        useMaterial3: true,
      ),
      home: const OcrDemoPage(),
    );
  }
}

class OcrDemoPage extends StatefulWidget {
  const OcrDemoPage({super.key});

  @override
  State<OcrDemoPage> createState() => _OcrDemoPageState();
}

class _OcrDemoPageState extends State<OcrDemoPage> {
  static const List<String> _testImageAssets = <String>[
    'assets/test_ocr/bob_ios_detection_issue.JPEG',
    'assets/test_ocr/heic_test.HEIC',
    'assets/test_ocr/mail_screenshot.jpeg',
    'assets/test_ocr/meme_ice_cream.jpeg',
    'assets/test_ocr/meme_love_you.jpeg',
    'assets/test_ocr/meme_perfect_couple.jpeg',
    'assets/test_ocr/meme_waking_up.jpeg',
    'assets/test_ocr/ocr_test.jpeg',
    'assets/test_ocr/payment_transactions.png',
    'assets/test_ocr/receipt_swiggy.jpg',
    'assets/test_ocr/screen_photos.jpeg',
    'assets/test_ocr/text_photos.jpeg',
  ];

  static bool get _isDesktop =>
      Platform.isLinux || Platform.isWindows || Platform.isMacOS;

  final ImagePicker _picker = ImagePicker();
  final MobileOcr _mobileOcr = MobileOcr();
  final TextDetectorController _textDetectorController =
      TextDetectorController();
  final Map<String, String> _cachedAssetPaths = <String, String>{};
  Directory? _assetCacheDirectory;
  String? _imagePath;
  bool _isPickingImage = false;
  bool _isCheckingHasText = false;
  bool? _lastHasTextResult;
  int? _currentTestImageIndex;
  bool _isLoadingTestImage = false;

  @override
  void dispose() {
    _textDetectorController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Mobile OCR'),
        actions: [
          if (_imagePath != null)
            AnimatedBuilder(
              animation: _textDetectorController,
              builder: (context, _) {
                final isReady =
                    _textDetectorController.hasSelectableText &&
                    !_textDetectorController.isProcessing;
                return IconButton(
                  tooltip: isReady
                      ? 'Select all recognized text'
                      : 'Detecting text...',
                  icon: const Icon(Icons.select_all_outlined),
                  onPressed: isReady
                      ? () => _handleSelectAllText(context)
                      : null,
                );
              },
            ),
          if (_imagePath != null)
            IconButton(
              tooltip: 'Clear image',
              icon: const Icon(Icons.close),
              onPressed: () {
                setState(() {
                  _imagePath = null;
                  _lastHasTextResult = null;
                  _isCheckingHasText = false;
                });
              },
            ),
        ],
      ),
      body: Column(
        children: [
          Expanded(child: _buildImageStage(context)),
          if (_lastHasTextResult != null)
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16),
              child: Align(
                alignment: Alignment.centerLeft,
                child: Text(
                  'hasText result: ${_lastHasTextResult!}',
                  style: Theme.of(context).textTheme.bodySmall,
                ),
              ),
            ),
          _buildActionBar(context),
        ],
      ),
    );
  }

  void _handleSelectAllText(BuildContext context) {
    final didSelect = _textDetectorController.selectAllText();
    if (!didSelect) {
      _showSnackBar(context, 'Text not ready yet');
      return;
    }
    _showSnackBar(context, 'Selected all recognized text');
  }

  Widget _buildPlaceholder(BuildContext context) {
    final theme = Theme.of(context);
    return Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(
            Icons.image_outlined,
            size: 96,
            color: theme.colorScheme.onSurface.withValues(alpha: 0.3),
          ),
          const SizedBox(height: 16),
          Text(
            'Pick an image to run OCR',
            style: theme.textTheme.titleMedium?.copyWith(
              color: theme.colorScheme.onSurface.withValues(alpha: 0.6),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildImageStage(BuildContext context) {
    final path = _imagePath;
    if (path == null) {
      return _buildPlaceholder(context);
    }

    return AnimatedBuilder(
      animation: _textDetectorController,
      builder: (context, _) {
        final showBaseImage =
            !_textDetectorController.hasSelectableText ||
            _textDetectorController.isProcessing;
        return Stack(
          fit: StackFit.expand,
          children: [
            if (showBaseImage)
              Positioned.fill(
                child: Image.file(File(path), fit: BoxFit.contain),
              ),
            TextDetectorWidget(
              key: ValueKey(path),
              imagePath: path,
              backgroundColor: Colors.transparent,
              debugMode: true,
              enableSelectionPreview: true,
              controller: _textDetectorController,
              onTextCopied: (text) => _showSnackBar(
                context,
                text.isEmpty
                    ? 'Copied empty text'
                    : 'Copied text (${text.length} chars)',
              ),
            ),
          ],
        );
      },
    );
  }

  Widget _buildActionBar(BuildContext context) {
    final hasImage = _imagePath != null;
    final isDesktop = _isDesktop;

    return SafeArea(
      top: false,
      child: Padding(
        padding: const EdgeInsets.fromLTRB(16, 12, 16, 20),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Row(
              children: [
                if (hasImage)
                  Expanded(
                    child: OutlinedButton.icon(
                      onPressed: _isCheckingHasText ? null : _checkHasText,
                      icon: const Icon(Icons.text_fields_outlined),
                      label: _isCheckingHasText
                          ? const Text('Checking...')
                          : const Text('hasText'),
                    ),
                  ),
                if (hasImage) const SizedBox(width: 12),
                Expanded(
                  child: OutlinedButton.icon(
                    onPressed: _isPickingImage
                        ? null
                        : () => _pickFromGallery(),
                    icon: const Icon(Icons.photo_library_outlined),
                    label: const Text('Gallery'),
                  ),
                ),
                if (!isDesktop) ...[
                  const SizedBox(width: 12),
                  Expanded(
                    child: FilledButton.icon(
                      onPressed: _isPickingImage
                          ? null
                          : () => _pickFromCamera(),
                      icon: const Icon(Icons.camera_alt_outlined),
                      label: const Text('Camera'),
                    ),
                  ),
                ],
              ],
            ),
            if (_testImageAssets.isNotEmpty) ...[
              const SizedBox(height: 12),
              Row(
                children: [
                  IconButton(
                    tooltip: 'Previous test image',
                    onPressed: _isLoadingTestImage
                        ? null
                        : () => _cycleTestImage(-1),
                    icon: const Icon(Icons.arrow_left),
                  ),
                  Expanded(
                    child: Center(
                      child: _isLoadingTestImage
                          ? const SizedBox(
                              width: 24,
                              height: 24,
                              child: CircularProgressIndicator(strokeWidth: 2),
                            )
                          : Text(
                              _currentTestImageIndex != null
                                  ? _formatAssetLabel(
                                      _testImageAssets[_currentTestImageIndex!],
                                    )
                                  : 'Tap arrows for test images',
                              style: Theme.of(context).textTheme.bodyMedium,
                              textAlign: TextAlign.center,
                            ),
                    ),
                  ),
                  IconButton(
                    tooltip: 'Next test image',
                    onPressed: _isLoadingTestImage
                        ? null
                        : () => _cycleTestImage(1),
                    icon: const Icon(Icons.arrow_right),
                  ),
                ],
              ),
            ],
          ],
        ),
      ),
    );
  }

  Future<void> _checkHasText() async {
    final path = _imagePath;
    if (path == null) {
      return;
    }

    setState(() {
      _isCheckingHasText = true;
      _lastHasTextResult = null;
    });

    try {
      debugPrint('Checking hasText for $path');
      final result = await _mobileOcr.hasText(imagePath: path);
      if (!mounted) return;
      setState(() {
        _lastHasTextResult = result;
      });
      _showSnackBar(context, 'hasText result: $result');
    } catch (error) {
      if (!mounted) return;
      debugPrint('hasText failed for $path: $error');
      _showSnackBar(context, 'hasText failed: $error');
    } finally {
      if (mounted) {
        setState(() {
          _isCheckingHasText = false;
        });
      }
    }
  }

  Future<void> _pickFromGallery() async {
    setState(() {
      _isPickingImage = true;
    });

    try {
      if (_isDesktop) {
        final result = await FilePicker.platform.pickFiles(
          type: FileType.image,
          allowMultiple: false,
        );
        if (result == null || result.files.isEmpty) {
          return;
        }
        final file = result.files.first;
        if (file.path == null) {
          return;
        }
        if (!mounted) return;
        setState(() {
          _imagePath = file.path;
          _lastHasTextResult = null;
          _isCheckingHasText = false;
        });
      } else {
        final file = await _picker.pickImage(source: ImageSource.gallery);
        if (file == null) {
          return;
        }
        if (!mounted) return;
        setState(() {
          _imagePath = file.path;
          _lastHasTextResult = null;
          _isCheckingHasText = false;
        });
      }
    } catch (error) {
      if (!mounted) return;
      _showSnackBar(context, 'Failed to pick image: $error');
    } finally {
      if (mounted) {
        setState(() {
          _isPickingImage = false;
        });
      }
    }
  }

  Future<void> _pickFromCamera() async {
    setState(() {
      _isPickingImage = true;
    });

    try {
      final file = await _picker.pickImage(source: ImageSource.camera);
      if (file == null) {
        return;
      }

      if (!mounted) return;
      setState(() {
        _imagePath = file.path;
        _lastHasTextResult = null;
        _isCheckingHasText = false;
      });
    } catch (error) {
      if (!mounted) return;
      _showSnackBar(context, 'Failed to pick image: $error');
    } finally {
      if (mounted) {
        setState(() {
          _isPickingImage = false;
        });
      }
    }
  }

  Future<void> _cycleTestImage(int direction) async {
    if (_testImageAssets.isEmpty || _isLoadingTestImage) {
      return;
    }
    final total = _testImageAssets.length;
    final currentIndex = _currentTestImageIndex;
    final nextIndex = currentIndex == null
        ? (direction >= 0 ? 0 : total - 1)
        : _wrapIndex(currentIndex + direction, total);
    setState(() {
      _isLoadingTestImage = true;
    });
    try {
      final assetPath = _testImageAssets[nextIndex];
      final filePath = await _prepareTestAsset(assetPath);
      if (!mounted) return;
      setState(() {
        _currentTestImageIndex = nextIndex;
        _imagePath = filePath;
        _lastHasTextResult = null;
        _isCheckingHasText = false;
      });
    } catch (error) {
      if (!mounted) return;
      _showSnackBar(context, 'Failed to load test image: $error');
    } finally {
      if (mounted) {
        setState(() {
          _isLoadingTestImage = false;
        });
      }
    }
  }

  Future<String> _prepareTestAsset(String assetPath) async {
    final cachedPath = _cachedAssetPaths[assetPath];
    if (cachedPath != null && await File(cachedPath).exists()) {
      return cachedPath;
    }
    final cacheDir = await _ensureAssetCacheDirectory();
    final fileName = assetPath.split('/').last;
    final filePath = '${cacheDir.path}${Platform.pathSeparator}$fileName';
    final file = File(filePath);
    final data = await rootBundle.load(assetPath);
    await file.writeAsBytes(
      data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes),
      flush: true,
    );
    _cachedAssetPaths[assetPath] = file.path;
    return file.path;
  }

  Future<Directory> _ensureAssetCacheDirectory() async {
    final existing = _assetCacheDirectory;
    if (existing != null) {
      return existing;
    }
    final directory = await Directory.systemTemp.createTemp(
      'mobile_ocr_example_assets_',
    );
    _assetCacheDirectory = directory;
    return directory;
  }

  String _formatAssetLabel(String assetPath) {
    final fileName = assetPath.split('/').last;
    final baseName = fileName.split('.').first;
    return baseName.replaceAll('_', ' ');
  }

  int _wrapIndex(int value, int length) {
    final mod = value % length;
    return mod < 0 ? mod + length : mod;
  }

  void _showSnackBar(BuildContext context, String message) {
    ScaffoldMessenger.of(context)
      ..removeCurrentSnackBar()
      ..showSnackBar(SnackBar(content: Text(message)));
  }
}
