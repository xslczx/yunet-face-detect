# yunet-face-detect

可复用的 Android 人脸检测库：封装 **YuNet** TFLite fp16 模型（`yunet_fp16.tflite`，0.25MB，离线推理，5 关键点），提供模型加载、预处理、推理、结果解析完整链路，支持 Kotlin / Java、同步 / 异步 / 协程调用，可作为 Gradle 模块、AAR 或本地 Maven 坐标集成。**EXIF 方向完整处理（含镜像 orientation 2/4/5/7）**。

---

## 1. 模型核实结论（官方一手来源，已在真机确认）

### 1.1 模型身份

| 项 | 值 |
|---|---|
| 原始模型 | YuNet（ShiqiYu/libfacedetection.train 的 yunet_n） |
| 转换版 | HuggingFace litert-community/YuNet-Face-LiteRT（litert-torch fp16） |
| 大小 / 参数量 | 256 KB / ~0.076M |
| SHA256 | `ced5f52bef6e76ad4a66d1055b2b404336ceafbae8eeac8fed6aa9c7b2e4d776` |
| 许可 | BSD-3-Clause |
| 出处 | Wu, Peng, Yu. *YuNet: A Tiny Millisecond-level Face Detector*. MIR 2023 |

**输入**：`[1, 3, 640, 640]` NCHW FLOAT32；**BGR**；像素 0-255 不归一化；官方评测管线为 letterbox + 双线性。

**输出**：12 个张量 `[1, N, C]`，官方导出顺序 `cls/obj/bbox/kps × stride 8/16/32`（N=6400/1600/400，共 8400 anchor）。
⚠️ 张量名为 `serving_default_output_{0..11}_output`，**不含角色关键字**，"按名称映射"会失败——库自动回退到官方导出顺序（`mappingFromNames=false`），实测无需手工配置 `outputIndices`。

**解码公式**（官方 postprocess/priors/codec）：

```
prior:  px = col*s, py = row*s        score = cls * obj（sigmoid 已在图内）
box:    cx = px + dx*s, cy = py + dy*s, w = exp(dw)*s, h = exp(dh)*s
kps:    landmark_i = (px + kps_x*s, py + kps_y*s)
后处理: 阈值过滤 -> 硬 NMS -> topK（官方默认 score 0.02 / nms 0.45，应用建议 0.3~0.9）
```

### 1.2 实测确认（真机 + 模拟器 API 37）

- landmark 顺序 `[左眼, 右眼, 鼻尖, 左嘴角, 右嘴角]` 目视核对成立；`eyeAngleDegrees()` 与照片倾斜一致。
- 官方管线组合显著更优（同一合照、阈值 0.6）：**LETTERBOX+BGR = 29 张** vs STRETCH 15 / RGB 9。建议保持默认。
- sigmoid 图内已含，`applySigmoidToScores(false)` 即可；实测分数分布 0.63~0.86。

### 1.3 适用场景

- **适合**：离线人脸检测；档案照/证件照裁剪与对齐（5 关键点）；隐私敏感（全本地）；包体受限（0.25MB）。
- **注意**：仅检测不识别（需另接 ArcFace 等）；>300px 特大脸建议下采样后检测；极端侧脸/重度遮挡不可靠；fp16 关键点噪声约 ±3°（对齐需加死区）；±90° 侧躺脸检不出，须保证 EXIF 转正。

---

## 2. 快速开始

### 2.1 放置模型

下载 `yunet_fp16.tflite`（256KB）放入 App 模块 `src/main/assets/`：
```
https://huggingface.co/litert-community/YuNet-Face-LiteRT/blob/main/yunet_fp16.tflite
```

### 2.2 三种集成方式

```groovy
// A. Gradle 模块（推荐）：拷入 yunetfacedetect/ 后
implementation project(':yunetfacedetect')

// B. AAR（flatDir 不传递依赖，需手动加 TFLite/协程/exifinterface）
implementation(name: 'yunetfacedetect-release', ext: 'aar')

// C. 本地 Maven（自动传递依赖）
implementation 'com.yunet.face:yunetfacedetect:1.0.0'
```

```bash
./gradlew :yunetfacedetect:assembleRelease      # 产出 release.aar (~89KB)
./gradlew :yunetfacedetect:publishToMavenLocal  # 发布 com.yunet.face:yunetfacedetect:1.0.0 到 ~/.m2
```

⚠️ mavenLocal 踩坑：若本机有全局 init 脚本注入镜像仓库（如阿里云），`PREFER_PROJECT` 模式下项目级仓库会覆盖 settings 级，`mavenLocal()` 被忽略报 `Could not find`。解法：把 `mavenLocal()` 放进**项目级** `repositories{}`，或 init 脚本里补一份。本仓库 `settings.gradle` 里注释掉的 `mavenLocal()` 就是该开关；demo 已实测换用 Maven 坐标可检出 29 张脸。

环境：守护进程声明 JDK 25（gradle-daemon-jvm.properties）。未装 openjdk@25 时可在 `~/.gradle/gradle.properties` 加
`org.gradle.java.installations.paths=/Applications/Android Studio.app/Contents/jbr/Contents/Home`。

### 2.3 上手（Kotlin / Java）

```kotlin
val detector = YuNetDetector.create(context, YuNetConfig.Builder()
    .useAsset("yunet_fp16.tflite")
    .scoreThreshold(0.6f)
    .build())

val faces = detector.detectSync(bitmap)          // 同步（勿在主线程）
detector.detectAsync(bitmap, object : DetectCallback {   // 异步（主线程回调）
    override fun onSuccess(faces: List<Face>) {}
    override fun onError(error: YuNetException) {}
})
val faces2 = detector.detect(bitmap)             // 协程挂起
val fileFaces = detector.detectFileSync("/sdcard/photo.jpg")  // 自动 EXIF 转正 + 采样解码
detector.close()
```

Java 侧：`YuNetDetector.create(...)` 带 `@JvmStatic`，回调用 `DetectCallback`，异常为非检查型 `YuNetException`（见 `JavaInteropDemo.java`）。

### 2.4 文件检测与坐标契约

```kotlin
// 文件存储像素坐标（既有语义，不施加 EXIF 变换）
val faces = detector.detectFileSync("/sdcard/photo.jpg")

// 两套坐标同时给出（推荐）：faces=存储像素坐标，uprightFaces=EXIF 转正后坐标
val r = detector.detectFileSyncDetailed("/sdcard/photo.jpg")
// r.rotationDegrees / r.mirrored / r.fileWidth×r.fileHeight / r.uprightWidth×r.uprightHeight
// 画到转正后的 ImageView 或用于裁剪/定位时，用 uprightFaces
```

⚠️ 坐标契约（最容易踩的坑）：文件照片的存储像素 ≠ 转正后展示图像。`detectFileSync` 返回**存储像素**坐标；需要转正坐标系时用 `detectFileSyncDetailed`（一次推理给两套结果）。镜像照片（orientation 2/4/5/7，前摄常见）由库统一处理：预处理**先水平镜像、后旋转**（androidx ExifInterface 约定），两套坐标均正确。

### 2.5 Demo App 与人脸裁剪

`app/` 模块（非发布产物）用于集成验证与回归。界面功能：

- **检测可视化**：人脸框 + 5 关键点 + 双眼连线 + 分数/倾角；配置面板（阈值滑块 0.05~0.95、拉伸/RGB/GPU 开关，改动下次检测生效并打印配置指纹）；调用方式下拉（detectAsync / detectSync / 协程）；自检与 Java 互操作。
- **人脸裁剪**（`ArchiveFaceCrop`，按钮驱动）：**始终以最近一次相册载入的原图为输入**（不是上一轮裁剪结果）；失败一律返回 null 并保留原图。流程：

  1. 采样解码（EXIF 转正，maxDim 2048）；
  2. **裁除四周纯色边条**（整行/列 RGB<30 判边条，每边 ≤20%——工具导出图的 pillarbox 黑条若不裁除，旋转后会斜切进出图）；
  3. 检出最大脸 → 按双眼连线角度校平（**3° 死区 + 25° 上限**，挡关键点噪声与误识别关键点）；
  4. 子图外扩旋转后重检，取景框在「照片覆盖范围」内收敛（逆旋转角点校验），**保留头顶空间、零补底**；重检丢失回退安全区，人脸框放不下退回更小图；
  5. 输出 3:4 JPEG，并打印 STEP 诊断日志（logcat `-s YuNetCrop`）。
- **裁剪参数面板**：滑杆 `头顶 / 下方 / 左右`（0.05 步长，×人脸框倍数）+ 三档预设 **紧凑(0.6/1.1/0.45)、标准(0.9/1.6/0.7)、宽松(1.2/2.2/1.0)** + 复位；点「人脸裁剪」时生效，日志打印当前参数。
- **`ArchiveFaceCrop.Params`**（API 可直接传）：`aspect`（宽高比 0.5~1.1）、`headroomScale`（头顶，0~2.5）、`chinroomScale`（下方，0~3.5）、`sideScale`（左右，0~2.5）、`tiltDeadZoneDeg`（0~10）、`tiltMaxDeg`（死区~45），init 带范围校验。

关键文件：

| 文件 | 作用 |
|---|---|
| `MainActivity.kt` | 配置面板、三种调用方式、裁剪参数面板、日志 |
| `FaceOverlayView.kt` | 框/关键点/倾角叠加绘制 |
| `ArchiveFaceCrop.kt` | 档案照裁剪流水线（见上） |
| `JavaInteropDemo.java` | 纯 Java 调用示例 |
| `src/androidTest/...YuNetDeviceTest.kt` + `JavaInteropDeviceTest.java` + `CropDebugProbeTest.kt` | 18 项设备端测试（测试样本在 `androidTest/assets/`，不随 App 打包） |

---

## 3. 配置项速查

| 方法 | 默认 | 说明 |
|---|---|---|
| `useAsset(name)` / `useFile(path)` / `useBytes(buf)` | 必选 | 模型来源，三选一 |
| `inputSize(w, h)` | 0（自适应） | 需正方形且 32 的倍数 |
| `inputLayout(...)` | AUTO | 本模型 NCHW |
| `channelOrder(...)` | BGR | 官方训练/评测为 BGR |
| `resizeMode(...)` | LETTERBOX | LETTERBOX=等比+补边；STRETCH=直接拉伸 |
| `padValue(v)` / `normalize(scale, mean)` | 0 / (1,0) | 补边灰度 / 归一化（官方不归一化） |
| `maxInputImageDim(d)` | 2048 | 输入位图最大边，防 OOM |
| `scoreThreshold(v)` / `nmsThreshold(v)` | 0.6 / 0.45 | 应用建议 0.3~0.9 |
| `topK(v)` / `nmsPreTopK(v)` | 0 / 5000 | NMS 后保留 / 候选上限 |
| `strides(vararg)` + `outputIndices(idx)` | 8,16,32 + null | 自定义需成对配置 |
| `applySigmoidToScores(b)` | false | 图内已含 sigmoid |
| `delegate(...)` | CPU | GPU/NNAPI，不可用自动回退 |
| `numThreads(n)` / `fallbackToCpu(b)` | 4 / true | 线程数 / 回退开关 |
| `autoRotateExif(b)` / `callbacksOnMainThread(b)` | true / true | 文件 EXIF 转正 / 回调线程 |

---

## 4. 错误处理

所有异常继承非检查异常 `YuNetException`：

| 异常 | 触发 | 建议 |
|---|---|---|
| `YuNetModelException` | 模型缺失/损坏、委托不可用且未允许回退 | 检查路径与依赖 |
| `YuNetIncompatibleModelException` | 形状/输出结构不符 | `YuNetDiagnostics.inspect()` 核对 |
| `YuNetInputException` | bitmap 回收、文件不存在 | 校验输入 |
| `YuNetInferenceException` | 推理运行期异常 | 看 cause，可切委托 |
| `YuNetClosedException` | close 后继续调用 | 重建检测器 |

异步接口不抛异常，错误经 `onError` 返回。

---

## 5. 自检与调优

```kotlin
Log.i("YuNet", YuNetDiagnostics.inspect(context, config))    // 结构自检
Log.i("YuNet", YuNetDiagnostics.selfTest(context, config, bitmap))  // 运行自检
```

**0 检出排查**：① 阈值降到 0.3（本模型最高分约 0.86）；② `inspect()` 核对 SHA256 与 12 路角色；③ STRETCH/RGB 只当"换了模型"的探针（实测会显著降低检出）；④ 方向——侧躺/倒置脸检不出，先 EXIF 转正。

**档案照裁剪**：直接参考 §2.5 与 `ArchiveFaceCrop.Params`；fp16 倾角噪声 ±3° 对应死区默认值。

---

## 6. 架构与扩展点

```
com.yunet.face
├── YuNetDetector          # 门面：create / detectSync / detectAsync / detect(挂起) / detectFile* / detectFileSyncDetailed / close
├── YuNetConfig            # 不可变配置 + Builder（含合法性校验）
├── Face                   # box + score + 5 关键点 + eyeAngleDegrees() + isLandmarksConsistent()
├── ExifTransform          # EXIF 旋转+镜像语义（先镜像、后旋转，androidx 约定）
├── FileDetectionResult    # detectFileSyncDetailed 结果：存储像素坐标 + 转正坐标
├── Callbacks.kt / YuNetException.kt
├── TfliteEngine           # 加载/自省/推理/线程安全/委托回退
├── Preprocess             # HARDWARE 转换、下采样、EXIF（镜像+旋转）、letterbox
├── CoordinateMapper       # net→竖直→EXIF 逆旋转→原图
├── decode/                # OutputDecoder 接口 + RawYuNetDecoder + Nms（换模型的扩展点）
└── YuNetDiagnostics       # 结构自检 + 运行自检
```

换 other 模型（OpenCV Zoo ONNX 版等）：实现 `OutputDecoder` + `outputIndices`/`inputLayout` 适配，其余链路零改动；接 RetinaFace/SCRFD 家族：实现 `FaceDetector` 接口。

**LiteRT 1.x 实测坑位**（`TfliteEngine.run()` 已按此实现）：多输出必须用 `runForMultipleInputsOutputs`（`Interpreter.run` 会把 Map 当单输出容器抛 `DataType error`）；输出容器用 direct `ByteBuffer` 而非扁平 `FloatArray`（形状校验会抛 `incompatible shape`），推理后 `asFloatBuffer().get()` 拷回。

---

## 7. 测试

```bash
gradle :yunetfacedetect:testDebugUnitTest      # JVM 单测 25 项（含 Robolectric 像素级预处理测试）
gradle :app:connectedDebugAndroidTest          # 设备端 20 项（app 模块）
```

JVM 单测覆盖 NMS、anchor-free 解码、sigmoid 开关、letterbox/stretch 映射、EXIF 逆旋转；Robolectric（native graphics）覆盖 `Preprocess` 八种 EXIF 方向组合的像素级转正（`PreprocessExifTest`）与坐标往返/关键点几何（`ExifMirrorTest`）。设备端（样本在 `app/src/androidTest/assets/`）覆盖：

| 测试 | 验证 |
|---|---|
| `modelStructureIsCompatible` | 12 路输出 / 8400 anchors / NCHW / FLOAT32 |
| `detectSyncFindsFacesOnSamplePhoto` | 29 张脸、分数降序、关键点齐备、倾角合理 |
| detectAsync / detect() 协程 / createAsync | 回调主线程、挂起函数、异步创建 |
| letterbox+stretch / BGR+RGB | 两种配置均检出（断言默认组合不劣） |
| 阈值 + topK | 高阈值检出不增、topK=2 生效 |
| `detectFileSyncHandlesLargeImageAndExifRotation` | 真实 EXIF orientation=6 转正（含无标签对照） |
| `detectFileSyncDetailedReportsBothCoordinateSystems` / `detectFileSyncMirroredExifRectifiesOrientation` | 两套坐标契约 / 镜像 orientation=2 转正 |
| 无效输入 / 灰图 / close 幂等 | 异常路径 |
| `warmUpPerformanceSmokeTest` | 预热耗时上界 |
| `JavaInteropDeviceTest` 4 项 | Java Builder/回调/异常/close |
| `CropDebugProbeTest` | 裁剪全链路（黑边裁除 + 校平 + 安全区收敛），打印 STEP 日志 |

---

## 8. 性能参考

| 平台 | 加速器 | 延迟 |
|---|---|---|
| Pixel 8a | GPU | ~4 ms @640×640 |
| Galaxy S26 | GPU / NPU | 3.3 / 1.85 ms |
| Raspberry Pi 5 | CPU | ~33.7 ms |

精度（YuNet_n，WIDER Face val）：AP_easy 0.892 / AP_medium 0.883 / AP_hard 0.811；可检人脸 10×10 ~ 300×300 px。

**本机实测**（2048×1150 合照 letterbox，CPU 4 线程，含预处理+推理+解码+NMS）：PLG110 真机 ~70 ms；Pixel_10a 模拟器 ~39 ms。阈值敏感：0.30 → 72 张 / 0.60 → 29 张 / 0.90 → 0 张。

---

## 9. 许可与出处

模型 BSD-3-Clause（litert-community 转换版）；训练代码 BSD-3-Clause（ShiqiYu/libfacedetection.train）。引用：`Wu W, Peng H, Yu S. YuNet: A tiny millisecond-level face detector. Machine Intelligence Research, 2023, 20(5): 656-665.`
