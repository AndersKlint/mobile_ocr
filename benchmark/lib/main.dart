import 'dart:io';

import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image_picker/image_picker.dart';
import 'package:mobile_ocr/mobile_ocr_plugin_method_channel.dart';
import 'package:mobile_ocr/mobile_ocr_plugin_dart.dart';
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

class BenchmarkResult {
  final int kotlinMs;
  final int dartMs;

  BenchmarkResult({required this.kotlinMs, required this.dartMs});

  String get winner {
    if (kotlinMs < dartMs) {
      final speedup = (dartMs / kotlinMs).toStringAsFixed(1);
      return 'Kotlin wins (${speedup}x faster)';
    } else if (dartMs < kotlinMs) {
      final speedup = (kotlinMs / dartMs).toStringAsFixed(1);
      return 'Dart wins (${speedup}x faster)';
    }
    return 'Tie';
  }
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'OCR Benchmark',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.indigo),
        useMaterial3: true,
      ),
      home: const OcrBenchmarkPage(),
    );
  }
}

class OcrBenchmarkPage extends StatefulWidget {
  const OcrBenchmarkPage({super.key});

  @override
  State<OcrBenchmarkPage> createState() => _OcrBenchmarkPageState();
}

class _OcrBenchmarkPageState extends State<OcrBenchmarkPage> {
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

  static const int _iterations = 20;
  static const int _totalCooldownSeconds = 120;

  static bool get _isDesktop =>
      Platform.isLinux || Platform.isWindows || Platform.isMacOS;
  static bool get _isAndroid => Platform.isAndroid;

  final ImagePicker _picker = ImagePicker();
  final TextDetectorController _textDetectorController =
      TextDetectorController();
  final Map<String, String> _cachedAssetPaths = <String, String>{};
  Directory? _assetCacheDirectory;

  MethodChannelMobileOcr? _kotlinOcr;
  DartMobileOcr? _dartOcr;
  bool _isInitializing = false;
  String? _initError;

  String? _imagePath;
  bool _isPickingImage = false;
  bool _isRunningBenchmark = false;
  bool _coolingDown = false;
  int _coolingDownSeconds = 0;
  BenchmarkResult? _lastResult;
  List<int>? _kotlinTimes;
  List<int>? _dartTimes;
  int? _currentTestImageIndex;
  bool _isLoadingTestImage = false;

  @override
  void initState() {
    super.initState();
    if (_isAndroid) {
      _initBenchmark();
    }
  }

  Future<void> _initBenchmark() async {
    setState(() {
      _isInitializing = true;
      _initError = null;
    });

    try {
      _kotlinOcr = MethodChannelMobileOcr();
      _dartOcr = DartMobileOcr();

      await _prepareModelsForAndroid();

      if (!mounted) return;
      setState(() {
        _isInitializing = false;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _isInitializing = false;
        _initError = e.toString();
      });
    }
  }

  Future<void> _prepareModelsForAndroid() async {
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

  @override
  void dispose() {
    _textDetectorController.dispose();
    _dartOcr?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('OCR Benchmark'),
        actions: [
          if (_imagePath != null && !_isAndroid)
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
              onPressed: _clearImage,
            ),
        ],
      ),
      body: Column(
        children: [
          Expanded(child: _buildImageStage(context)),
          if (_isAndroid) _buildBenchmarkPanel(context),
          _buildActionBar(context),
        ],
      ),
    );
  }

  void _clearImage() {
    setState(() {
      _imagePath = null;
      _lastResult = null;
      _kotlinTimes = null;
      _dartTimes = null;
      _currentTestImageIndex = null;
      _coolingDown = false;
      _coolingDownSeconds = 0;
    });
  }

  void _handleSelectAllText(BuildContext context) {
    final didSelect = _textDetectorController.selectAllText();
    if (!didSelect) {
      _showSnackBar(context, 'Text not ready yet');
      return;
    }
    _showSnackBar(context, 'Selected all recognized text');
  }

  Widget _buildBenchmarkPanel(BuildContext context) {
    final theme = Theme.of(context);

    if (_isInitializing) {
      return Container(
        width: double.infinity,
        padding: const EdgeInsets.all(16),
        color: theme.colorScheme.surfaceContainerHighest,
        child: Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const SizedBox(
              width: 20,
              height: 20,
              child: CircularProgressIndicator(strokeWidth: 2),
            ),
            const SizedBox(width: 12),
            Text(
              'Initializing OCR engines...',
              style: theme.textTheme.bodyMedium,
            ),
          ],
        ),
      );
    }

    if (_initError != null) {
      return Container(
        width: double.infinity,
        padding: const EdgeInsets.all(16),
        color: theme.colorScheme.errorContainer,
        child: Text(
          'Init error: $_initError',
          style: theme.textTheme.bodyMedium?.copyWith(
            color: theme.colorScheme.onErrorContainer,
          ),
        ),
      );
    }

    if (_isRunningBenchmark && !_coolingDown) {
      return Container(
        width: double.infinity,
        padding: const EdgeInsets.all(16),
        color: theme.colorScheme.surfaceContainerHighest,
        child: Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const SizedBox(
              width: 20,
              height: 20,
              child: CircularProgressIndicator(strokeWidth: 2),
            ),
            const SizedBox(width: 12),
            Text(
              'Running benchmark ($_iterations iterations each)...',
              style: theme.textTheme.bodyMedium,
            ),
          ],
        ),
      );
    }

    if (_coolingDown) {
      return Container(
        width: double.infinity,
        padding: const EdgeInsets.all(16),
        color: theme.colorScheme.surfaceContainerHighest,
        child: Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const SizedBox(
              width: 20,
              height: 20,
              child: CircularProgressIndicator(strokeWidth: 2),
            ),
            const SizedBox(width: 12),
            Text(
              'Cooling down... ${_coolingDownSeconds}s remaining',
              style: theme.textTheme.bodyMedium,
            ),
          ],
        ),
      );
    }

    if (_lastResult == null || _kotlinTimes == null || _dartTimes == null) {
      return const SizedBox.shrink();
    }

    final kotlinAvg =
        _kotlinTimes!.reduce((a, b) => a + b) ~/ _kotlinTimes!.length;
    final dartAvg = _dartTimes!.reduce((a, b) => a + b) ~/ _dartTimes!.length;

    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: theme.colorScheme.surfaceContainerHighest,
        border: Border(
          top: BorderSide(color: theme.colorScheme.outlineVariant),
        ),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisSize: MainAxisSize.min,
        children: [
          Text(
            'Benchmark Results ($_iterations iterations)',
            style: theme.textTheme.titleSmall?.copyWith(
              fontWeight: FontWeight.bold,
            ),
          ),
          const SizedBox(height: 12),
          _buildResultRow(
            context,
            'Kotlin (native)',
            _kotlinTimes!,
            kotlinAvg,
            Colors.green,
          ),
          const SizedBox(height: 8),
          _buildResultRow(context, 'Dart', _dartTimes!, dartAvg, Colors.blue),
          const SizedBox(height: 12),
          Container(
            width: double.infinity,
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
            decoration: BoxDecoration(
              color: theme.colorScheme.primaryContainer,
              borderRadius: BorderRadius.circular(8),
            ),
            child: Text(
              _lastResult!.winner,
              style: theme.textTheme.titleMedium?.copyWith(
                fontWeight: FontWeight.bold,
                color: theme.colorScheme.onPrimaryContainer,
              ),
              textAlign: TextAlign.center,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildResultRow(
    BuildContext context,
    String label,
    List<int> times,
    int avg,
    Color color,
  ) {
    final theme = Theme.of(context);
    final min = times.reduce((a, b) => a < b ? a : b);
    final max = times.reduce((a, b) => a > b ? a : b);

    return Row(
      children: [
        Container(
          width: 4,
          height: 36,
          decoration: BoxDecoration(
            color: color,
            borderRadius: BorderRadius.circular(2),
          ),
        ),
        const SizedBox(width: 12),
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                label,
                style: theme.textTheme.bodySmall?.copyWith(
                  fontWeight: FontWeight.w600,
                ),
              ),
              Text(
                'avg: ${avg}ms | min: ${min}ms | max: ${max}ms',
                style: theme.textTheme.bodyMedium?.copyWith(
                  fontFeatures: [const FontFeature.tabularFigures()],
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }

  Widget _buildPlaceholder(BuildContext context) {
    final theme = Theme.of(context);
    return Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(
            Icons.speed_outlined,
            size: 96,
            color: theme.colorScheme.onSurface.withValues(alpha: 0.3),
          ),
          const SizedBox(height: 16),
          Text(
            'Pick an image to run OCR benchmark',
            style: theme.textTheme.titleMedium?.copyWith(
              color: theme.colorScheme.onSurface.withValues(alpha: 0.6),
            ),
          ),
          if (_isAndroid) ...[
            const SizedBox(height: 8),
            Text(
              'Comparing Kotlin vs Dart implementations',
              style: theme.textTheme.bodySmall?.copyWith(
                color: theme.colorScheme.onSurface.withValues(alpha: 0.5),
              ),
            ),
          ],
        ],
      ),
    );
  }

  Widget _buildImageStage(BuildContext context) {
    final path = _imagePath;
    if (path == null) {
      return _buildPlaceholder(context);
    }

    if (_isAndroid) {
      return Stack(
        fit: StackFit.expand,
        children: [
          Positioned.fill(child: Image.file(File(path), fit: BoxFit.contain)),
          if (_isRunningBenchmark || _coolingDown)
            Container(
              color: Colors.black45,
              child: Center(
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    const CircularProgressIndicator(),
                    const SizedBox(height: 16),
                    Text(
                      _coolingDown
                          ? 'Cooling down... ${_coolingDownSeconds}s'
                          : 'Running benchmark...',
                      style: Theme.of(
                        context,
                      ).textTheme.titleMedium?.copyWith(color: Colors.white),
                    ),
                  ],
                ),
              ),
            ),
        ],
      );
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
                Expanded(
                  child: OutlinedButton.icon(
                    onPressed: _isPickingImage || _isRunningBenchmark
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
                      onPressed: _isPickingImage || _isRunningBenchmark
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
                    onPressed: _isLoadingTestImage || _isRunningBenchmark
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
                    onPressed: _isLoadingTestImage || _isRunningBenchmark
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

  Future<void> _runBenchmark(String imagePath) async {
    if (!_isAndroid || _kotlinOcr == null || _dartOcr == null) return;

    setState(() {
      _isRunningBenchmark = true;
      _lastResult = null;
      _kotlinTimes = null;
      _dartTimes = null;
    });

    try {
      final kotlinTimes = <int>[];
      final dartTimes = <int>[];

      for (int i = 0; i < _iterations; i++) {
        final kotlinStopwatch = Stopwatch()..start();
        await _kotlinOcr!.detectText(imagePath: imagePath);
        kotlinStopwatch.stop();
        kotlinTimes.add(kotlinStopwatch.elapsedMilliseconds);

        await Future.delayed(const Duration(milliseconds: 500));
      }

      setState(() {
        _coolingDown = true;
        _coolingDownSeconds = _totalCooldownSeconds;
      });

      for (int i = _coolingDownSeconds; i > 0; i--) {
        if (!mounted) return;
        setState(() {
          _coolingDownSeconds = i;
        });
        await Future.delayed(const Duration(seconds: 1));
      }

      setState(() {
        _coolingDown = false;
      });

      for (int i = 0; i < _iterations; i++) {
        final dartStopwatch = Stopwatch()..start();
        await _dartOcr!.detectText(imagePath: imagePath);
        dartStopwatch.stop();
        dartTimes.add(dartStopwatch.elapsedMilliseconds);

        await Future.delayed(const Duration(milliseconds: 500));
      }

      if (!mounted) return;

      final kotlinAvg =
          kotlinTimes.reduce((a, b) => a + b) ~/ kotlinTimes.length;
      final dartAvg = dartTimes.reduce((a, b) => a + b) ~/ dartTimes.length;

      setState(() {
        _kotlinTimes = kotlinTimes;
        _dartTimes = dartTimes;
        _lastResult = BenchmarkResult(kotlinMs: kotlinAvg, dartMs: dartAvg);
        _isRunningBenchmark = false;
      });

      debugPrint('Benchmark: Kotlin=$kotlinTimes, Dart=$dartTimes');
      debugPrint(_lastResult!.winner);
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _isRunningBenchmark = false;
      });
      _showSnackBar(context, 'Benchmark failed: $e');
      debugPrint('Benchmark error: $e');
    }
  }

  Future<void> _pickFromGallery() async {
    setState(() {
      _isPickingImage = true;
    });

    try {
      String? path;
      if (_isDesktop) {
        final result = await FilePicker.platform.pickFiles(
          type: FileType.image,
          allowMultiple: false,
        );
        if (result == null || result.files.isEmpty) return;
        path = result.files.first.path;
      } else {
        final file = await _picker.pickImage(source: ImageSource.gallery);
        path = file?.path;
      }

      if (path == null || !mounted) return;

      setState(() {
        _imagePath = path;
        _currentTestImageIndex = null;
      });

      if (_isAndroid) {
        await _runBenchmark(path);
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
      if (file == null || !mounted) return;

      setState(() {
        _imagePath = file.path;
        _currentTestImageIndex = null;
      });

      if (_isAndroid) {
        await _runBenchmark(file.path);
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

  Future<void> _cycleTestImage(int direction) async {
    if (_testImageAssets.isEmpty || _isLoadingTestImage) return;

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
      });

      if (_isAndroid) {
        await _runBenchmark(filePath);
      }
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
    if (existing != null) return existing;
    final directory = await Directory.systemTemp.createTemp(
      'mobile_ocr_benchmark_assets_',
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
