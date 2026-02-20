import 'dart:ffi' as ffi;
import 'dart:typed_data';

import 'package:ffi/ffi.dart';
import 'package:onnxruntime_v2/onnxruntime_v2.dart';
import 'package:onnxruntime_v2/src/bindings/onnxruntime_bindings_generated.dart'
    as bg;

class FastTensorReader {
  static Float32List? asFloat32List(OrtValue output) {
    final ortPtr = output.ptr;

    final infoPtrPtr = calloc<ffi.Pointer<bg.OrtTensorTypeAndShapeInfo>>();
    final statusPtr = OrtEnv.instance.ortApiPtr.ref.GetTensorTypeAndShape
        .asFunction<
          bg.OrtStatusPtr Function(
            ffi.Pointer<bg.OrtValue>,
            ffi.Pointer<ffi.Pointer<bg.OrtTensorTypeAndShapeInfo>>,
          )
        >()(ortPtr, infoPtrPtr);

    if (statusPtr != ffi.nullptr) {
      calloc.free(infoPtrPtr);
      return null;
    }

    final infoPtr = infoPtrPtr.value;
    calloc.free(infoPtrPtr);

    final countPtr = calloc<ffi.Size>();
    var status = OrtEnv.instance.ortApiPtr.ref.GetTensorShapeElementCount
        .asFunction<
          bg.OrtStatusPtr Function(
            ffi.Pointer<bg.OrtTensorTypeAndShapeInfo>,
            ffi.Pointer<ffi.Size>,
          )
        >()(infoPtr, countPtr);

    if (status != ffi.nullptr) {
      OrtEnv.instance.ortApiPtr.ref.ReleaseTensorTypeAndShapeInfo
          .asFunction<
            void Function(ffi.Pointer<bg.OrtTensorTypeAndShapeInfo>)
          >()(infoPtr);
      calloc.free(countPtr);
      return null;
    }

    final elementCount = countPtr.value;
    calloc.free(countPtr);

    if (elementCount <= 0 || elementCount > 1000000000) {
      OrtEnv.instance.ortApiPtr.ref.ReleaseTensorTypeAndShapeInfo
          .asFunction<
            void Function(ffi.Pointer<bg.OrtTensorTypeAndShapeInfo>)
          >()(infoPtr);
      return null;
    }

    final dataPtrPtr = calloc<ffi.Pointer<ffi.Float>>();
    status = OrtEnv.instance.ortApiPtr.ref.GetTensorMutableData
        .asFunction<
          bg.OrtStatusPtr Function(
            ffi.Pointer<bg.OrtValue>,
            ffi.Pointer<ffi.Pointer<ffi.Void>>,
          )
        >()(ortPtr, dataPtrPtr.cast());

    OrtEnv.instance.ortApiPtr.ref.ReleaseTensorTypeAndShapeInfo
        .asFunction<void Function(ffi.Pointer<bg.OrtTensorTypeAndShapeInfo>)>()(
      infoPtr,
    );

    if (status != ffi.nullptr) {
      calloc.free(dataPtrPtr);
      return null;
    }

    final dataPtr = dataPtrPtr.value;
    calloc.free(dataPtrPtr);

    return dataPtr.asTypedList(elementCount);
  }

  static List<int> getShape(OrtValue output) {
    final ortPtr = output.ptr;

    final infoPtrPtr = calloc<ffi.Pointer<bg.OrtTensorTypeAndShapeInfo>>();
    final statusPtr = OrtEnv.instance.ortApiPtr.ref.GetTensorTypeAndShape
        .asFunction<
          bg.OrtStatusPtr Function(
            ffi.Pointer<bg.OrtValue>,
            ffi.Pointer<ffi.Pointer<bg.OrtTensorTypeAndShapeInfo>>,
          )
        >()(ortPtr, infoPtrPtr);

    if (statusPtr != ffi.nullptr) {
      calloc.free(infoPtrPtr);
      return [];
    }

    final infoPtr = infoPtrPtr.value;
    calloc.free(infoPtrPtr);

    final dimsCountPtr = calloc<ffi.Size>();
    var status = OrtEnv.instance.ortApiPtr.ref.GetDimensionsCount
        .asFunction<
          bg.OrtStatusPtr Function(
            ffi.Pointer<bg.OrtTensorTypeAndShapeInfo>,
            ffi.Pointer<ffi.Size>,
          )
        >()(infoPtr, dimsCountPtr);

    if (status != ffi.nullptr) {
      OrtEnv.instance.ortApiPtr.ref.ReleaseTensorTypeAndShapeInfo
          .asFunction<
            void Function(ffi.Pointer<bg.OrtTensorTypeAndShapeInfo>)
          >()(infoPtr);
      calloc.free(dimsCountPtr);
      return [];
    }

    final dimsCount = dimsCountPtr.value;
    calloc.free(dimsCountPtr);

    final dimsPtr = calloc<ffi.Int64>(dimsCount);
    status = OrtEnv.instance.ortApiPtr.ref.GetDimensions
        .asFunction<
          bg.OrtStatusPtr Function(
            ffi.Pointer<bg.OrtTensorTypeAndShapeInfo>,
            ffi.Pointer<ffi.Int64>,
            int,
          )
        >()(infoPtr, dimsPtr, dimsCount);

    OrtEnv.instance.ortApiPtr.ref.ReleaseTensorTypeAndShapeInfo
        .asFunction<void Function(ffi.Pointer<bg.OrtTensorTypeAndShapeInfo>)>()(
      infoPtr,
    );

    if (status != ffi.nullptr) {
      calloc.free(dimsPtr);
      return [];
    }

    final shape = <int>[];
    for (int i = 0; i < dimsCount; i++) {
      shape.add(dimsPtr[i]);
    }
    calloc.free(dimsPtr);

    return shape;
  }
}
