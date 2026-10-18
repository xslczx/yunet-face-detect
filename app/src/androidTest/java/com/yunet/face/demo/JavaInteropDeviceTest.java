package com.yunet.face.demo;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.util.Log;

import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.platform.app.InstrumentationRegistry;

import com.yunet.face.DetectCallback;
import com.yunet.face.Face;
import com.yunet.face.YuNetClosedException;
import com.yunet.face.YuNetConfig;
import com.yunet.face.YuNetDetector;
import com.yunet.face.YuNetException;
import com.yunet.face.YuNetModelException;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;

import java.io.InputStream;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Java 侧互操作测试：验证 Kotlin 库在 Java 调用点上的可用性
 * （Builder / @JvmStatic create / 接口回调 / 非检查异常体系 / close 幂等）。
 *
 * 与 Kotlin 的 {@link YuNetDeviceTest} 互补：这里专测“从 Java 调用”这条路径。
 */
@RunWith(AndroidJUnit4.class)
public class JavaInteropDeviceTest {

    private static final String TAG = "JavaInteropDeviceTest";
    private static final String MODEL = "yunet_fp16.tflite";
    private static final String SAMPLE = "sample/largest_selfie.jpg";
    private static final long TIMEOUT_S = 60L;

    private Context ctx;
    private Bitmap sample;
    private YuNetDetector detector;

    @Before
    public void setUp() throws Exception {
        ctx = InstrumentationRegistry.getInstrumentation().getTargetContext();
        // 示例图只随 androidTest APK 打包（app/src/androidTest/assets/），从 instrumentation 上下文读取
        try (InputStream in = InstrumentationRegistry.getInstrumentation().getContext()
                .getAssets().open(SAMPLE)) {
            sample = BitmapFactory.decodeStream(in);
        }
        assertNotNull("androidTest assets/" + SAMPLE + " 解码失败", sample);
    }

    @After
    public void tearDown() {
        if (detector != null) {
            detector.close();
            detector = null;
        }
        if (sample != null) {
            sample.recycle();
            sample = null;
        }
    }

    private YuNetConfig config() {
        return new YuNetConfig.Builder()
                .useAsset(MODEL)
                .scoreThreshold(0.6f)
                .topK(100)
                .build();
    }

    @Test
    public void createAndDetectSyncFromJava() {
        detector = YuNetDetector.create(ctx, config());
        Log.i(TAG, "modelInfo=\n" + detector.getModelInfo());

        List<Face> faces = detector.detectSync(sample);
        Log.i(TAG, "detectSync 检出 " + faces.size());
        assertTrue("Java 侧同步检测应有检出", faces.size() >= 3);

        Face first = faces.get(0);
        Log.i(TAG, "first=" + first);
        assertTrue("score 应在 (0,1]", first.getScore() > 0f && first.getScore() <= 1.0001f);
        // width/height 是 @JvmField，Java 侧按字段访问
        assertTrue("框宽高应为正", first.width > 0f && first.height > 0f);
        assertNotNull("关键点应可用", first.getLandmarks());
        assertEquals("关键点应为 10 个 float", 10, first.getLandmarks().length);
        assertNotNull("左眼关键点应可用", first.getLeftEye());
        assertNotNull("box 应可用", first.getBox());
        assertTrue("倾角应为有限值", !Float.isNaN(first.eyeAngleDegrees()));
    }

    @Test
    public void detectAsyncFromJava() throws Exception {
        detector = YuNetDetector.create(ctx, config());
        final AtomicReference<List<Face>> facesRef = new AtomicReference<>();
        final AtomicReference<YuNetException> errorRef = new AtomicReference<>();
        final CountDownLatch latch = new CountDownLatch(1);

        detector.detectAsync(sample, new DetectCallback() {
            @Override
            public void onSuccess(List<Face> faces) {
                facesRef.set(faces);
                latch.countDown();
            }

            @Override
            public void onError(YuNetException error) {
                errorRef.set(error);
                latch.countDown();
            }
        });

        assertTrue("Java 异步回调超时", latch.await(TIMEOUT_S, TimeUnit.SECONDS));
        assertNull("Java 异步不应报错: " + errorRef.get(), errorRef.get());
        assertNotNull(facesRef.get());
        assertTrue("Java 异步检测应有检出", !facesRef.get().isEmpty());
    }

    @Test
    public void javaCallsAfterCloseThrowUncheckedException() {
        detector = YuNetDetector.create(ctx, config());
        detector.close();
        assertTrue(detector.isClosed());

        try {
            detector.detectSync(sample);
            fail("close() 后调用应抛 YuNetClosedException");
        } catch (YuNetClosedException e) {
            // 非检查异常：Java 编译期无需声明 throws，捕获仅为断言
            Log.i(TAG, "正确捕获 " + e.getClass().getSimpleName() + ": " + e.getMessage());
        } catch (RuntimeException e) {
            fail("应为 YuNetClosedException，实际 " + e.getClass().getName());
        }
    }

    @Test
    public void invalidModelAssetThrowsYuNetModelException() {
        YuNetConfig bad = new YuNetConfig.Builder()
                .useAsset("not-exist-model.tflite")
                .build();
        try {
            YuNetDetector.create(ctx, bad);
            fail("缺失模型应抛 YuNetException");
        } catch (YuNetException e) {
            Log.i(TAG, "缺失模型正确抛出 " + e.getClass().getSimpleName() + ": " + e.getMessage());
            assertTrue("应为 YuNetModelException", e instanceof YuNetModelException);
        }
    }
}
