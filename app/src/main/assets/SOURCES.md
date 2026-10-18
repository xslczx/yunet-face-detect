# demo assets 来源

| 文件 | 来源 | 许可 / 说明 |
|---|---|---|
| `yunet_fp16.tflite` | https://huggingface.co/litert-community/YuNet-Face-LiteRT (`yunet_fp16.tflite`) | BSD-3-Clause。sha256 `ced5f52bef6e76ad4a66d1055b2b404336ceafbae8eeac8fed6aa9c7b2e4d776`，256 KB。国内可换用镜像：`https://hf-mirror.com/litert-community/YuNet-Face-LiteRT/resolve/main/yunet_fp16.tflite` |

示例合照 `app/src/androidTest/assets/sample/largest_selfie.jpg`（https://github.com/opencv/opencv_zoo → `models/face_detection_yunet/example_outputs/largest_selfie.jpg`，Apache-2.0，OpenCV YuNet 官方示例图，多人脸正样本 2048x1150）**只随 androidTest APK 打包**，供设备端集成测试使用，App 包不含示例图。

`yunet_fp16.tflite` 是唯一的运行必需文件（放在 App 模块的 `src/main/assets/` 即可）；接入自己的 App 时只需带上它。
