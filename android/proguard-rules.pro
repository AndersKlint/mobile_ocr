# Keep mobile_ocr plugin classes
-keep class io.ente.mobile_ocr.** { *; }
-dontwarn io.ente.mobile_ocr.**

# Keep ONNX Runtime classes
-keep class ai.onnxruntime.** { *; }
-dontwarn ai.onnxruntime.**

# Keep native methods
-keepclasseswithmembernames class * {
    native <methods>;
}
