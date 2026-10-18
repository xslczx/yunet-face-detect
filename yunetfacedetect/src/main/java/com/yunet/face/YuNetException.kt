package com.yunet.face

/**
 * 库内所有异常的基类（非检查异常）。Java 调用方可 catch YuNetException 统一处理。
 */
open class YuNetException(message: String, cause: Throwable? = null) : RuntimeException(message, cause)

/** 模型加载/初始化失败：文件缺失、格式错误、委托不可用等 */
class YuNetModelException(message: String, cause: Throwable? = null) : YuNetException(message, cause)

/** 模型结构与解码器不兼容：输入形状/dtype、输出数量/通道数不符合 12 路 YuNet 约定 */
class YuNetIncompatibleModelException(message: String, cause: Throwable? = null) : YuNetException(message, cause)

/** 调用方输入非法：bitmap 为空/已回收、文件不存在或无法解码等 */
class YuNetInputException(message: String, cause: Throwable? = null) : YuNetException(message, cause)

/** 推理执行期异常 */
class YuNetInferenceException(message: String, cause: Throwable? = null) : YuNetException(message, cause)

/** 检测器已关闭后继续调用 */
class YuNetClosedException(message: String) : YuNetException(message)
