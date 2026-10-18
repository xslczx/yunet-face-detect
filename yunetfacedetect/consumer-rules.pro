# TFLite 通过 JNI 访问，且 GPU 委托由本库反射加载，需对调用方保留
-keep class org.tensorflow.lite.** { *; }
-keep class org.tensorflow.lite.gpu.GpuDelegate { *; }
-dontwarn org.tensorflow.lite.gpu.**
