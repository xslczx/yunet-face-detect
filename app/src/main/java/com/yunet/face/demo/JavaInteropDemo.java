package com.yunet.face.demo;

import android.content.Context;
import android.graphics.Bitmap;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;

import com.yunet.face.DetectCallback;
import com.yunet.face.Face;
import com.yunet.face.YuNetConfig;
import com.yunet.face.YuNetDetector;
import com.yunet.face.YuNetException;

import java.util.List;

/**
 * 纯 Java 调用示例：验证库的 Java 互操作性（Builder / create / detectAsync / 非检查异常 / close）。
 *
 * 与 Kotlin 侧不同，这里不使用协程，只用「后台线程 + 回调」，对应 README 2.5 的 Java 用法。
 */
public final class JavaInteropDemo {

    private static final Handler MAIN = new Handler(Looper.getMainLooper());

    private JavaInteropDemo() {
    }

    /** 结果回调，始终在主线程调用 */
    public interface ResultListener {
        void onResult(String summary, int faceCount);
    }

    /** 统一把回调切到主线程（创建失败的分支在后台线程触发） */
    private static void post(Runnable r) {
        if (Looper.myLooper() == Looper.getMainLooper()) {
            r.run();
        } else {
            MAIN.post(r);
        }
    }

    /**
     * 后台线程创建检测器 -> detectAsync 异步检测 -> 主线程回调结果。
     *
     * @param context  任意 Context（内部会转为 applicationContext）
     * @param bitmap   待检测图片，勿在回调前回收
     * @param listener 结果回调（主线程）
     */
    public static void detect(final Context context, final Bitmap bitmap, final ResultListener listener) {
        new Thread(new Runnable() {
            @Override
            public void run() {
                final YuNetConfig config = new YuNetConfig.Builder()
                        .useAsset("yunet_fp16.tflite")
                        .scoreThreshold(0.6f)
                        .topK(50)
                        .build();

                final YuNetDetector detector;
                try {
                    detector = YuNetDetector.create(context, config);
                } catch (Throwable t) {
                    // YuNetException 是非检查异常，Java 侧无需 try/catch 也能编译；此处演示统一处理
                    final String summary = "Java: 创建检测器失败 " + describe(t);
                    post(new Runnable() {
                        @Override
                        public void run() {
                            listener.onResult(summary, -1);
                        }
                    });
                    return;
                }

                final long t0 = System.currentTimeMillis();
                detector.detectAsync(bitmap, new DetectCallback() {
                    @Override
                    public void onSuccess(List<Face> faces) {
                        StringBuilder sb = new StringBuilder();
                        sb.append("Java: detectAsync 成功，耗时 ")
                                .append(System.currentTimeMillis() - t0).append("ms，检出 ")
                                .append(faces.size()).append(" 张人脸");
                        for (int i = 0; i < faces.size() && i < 5; i++) {
                            Face f = faces.get(i);
                            sb.append(String.format("%n   #%d score=%.3f box=[%.0f,%.0f,%.0f,%.0f] eye=%.1f°",
                                    i, f.getScore(),
                                    f.getBox().left, f.getBox().top, f.getBox().right, f.getBox().bottom,
                                    f.eyeAngleDegrees()));
                        }
                        final String summary = sb.toString();
                        final int count = faces.size();
                        detector.close();
                        post(new Runnable() {
                            @Override
                            public void run() {
                                listener.onResult(summary, count);
                            }
                        });
                    }

                    @Override
                    public void onError(YuNetException error) {
                        final String summary = "Java: 检测失败 " + describe(error);
                        detector.close();
                        post(new Runnable() {
                            @Override
                            public void run() {
                                listener.onResult(summary, -1);
                            }
                        });
                    }
                });
            }
        }, "java-interop-demo").start();
    }

    private static String describe(Throwable t) {
        Log.w("JavaInteropDemo", "调用失败", t);
        return t.getClass().getSimpleName() + ": " + t.getMessage();
    }
}
