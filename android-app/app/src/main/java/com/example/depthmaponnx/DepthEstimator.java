package com.example.depthmaponnx;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Color;

import com.microsoft.onnxruntime.OnnxTensor;
import com.microsoft.onnxruntime.OrtEnvironment;
import com.microsoft.onnxruntime.OrtException;
import com.microsoft.onnxruntime.OrtSession;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.FloatBuffer;

/**
 * Runs depth estimation using an ONNX model.
 * The model file "depth_anything_v2_large.onnx" should be placed in the assets folder.
 */
public class DepthEstimator {
    private final OrtEnvironment env;
    private final OrtSession session;

    public DepthEstimator(AssetManager am) throws IOException, OrtException {
        env = OrtEnvironment.getEnvironment();
        byte[] model = readAsset(am, "depth_anything_v2_large.onnx");
        session = env.createSession(model, new OrtSession.SessionOptions());
    }

    private byte[] readAsset(AssetManager am, String name) throws IOException {
        InputStream is = am.open(name);
        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        byte[] buffer = new byte[4096];
        int read;
        while ((read = is.read(buffer)) != -1) {
            bos.write(buffer, 0, read);
        }
        is.close();
        return bos.toByteArray();
    }

    public float[][] estimateDepth(Bitmap bitmap) throws OrtException {
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        Bitmap resized = Bitmap.createScaledBitmap(bitmap, 384, 384, true);
        float[] inputData = new float[1 * 3 * 384 * 384];
        int[] pixels = new int[384 * 384];
        resized.getPixels(pixels, 0, 384, 0, 0, 384, 384);
        for (int y = 0; y < 384; y++) {
            for (int x = 0; x < 384; x++) {
                int color = pixels[y * 384 + x];
                float r = (Color.red(color) / 255f);
                float g = (Color.green(color) / 255f);
                float b = (Color.blue(color) / 255f);
                int base = y * 384 + x;
                inputData[base] = r;
                inputData[base + 384 * 384] = g;
                inputData[base + 2 * 384 * 384] = b;
            }
        }
        FloatBuffer buffer = FloatBuffer.wrap(inputData);
        OnnxTensor tensor = OnnxTensor.createTensor(env, buffer, new long[]{1, 3, 384, 384});
        OrtSession.Result result = session.run(java.util.Collections.singletonMap(session.getInputNames().iterator().next(), tensor));
        float[] depthRaw = (float[]) result.get(0).getValue();
        float[][] depth = new float[height][width];
        // Resize back to original size
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int srcX = x * 384 / width;
                int srcY = y * 384 / height;
                float v = depthRaw[srcY * 384 + srcX];
                depth[y][x] = v;
            }
        }
        // Normalize
        float min = Float.MAX_VALUE;
        float max = -Float.MAX_VALUE;
        for (float[] row : depth) {
            for (float v : row) {
                if (v < min) min = v;
                if (v > max) max = v;
            }
        }
        float range = max - min;
        if (range > 0) {
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    depth[y][x] = (depth[y][x] - min) / range;
                }
            }
        }
        return depth;
    }

    public static Bitmap toGrayBitmap(float[][] depth) {
        int h = depth.length;
        int w = depth[0].length;
        Bitmap bmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888);
        int[] pixels = new int[w * h];
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int v = (int) (depth[y][x] * 255f);
                v = Math.max(0, Math.min(255, v));
                pixels[y * w + x] = Color.rgb(v, v, v);
            }
        }
        bmp.setPixels(pixels, 0, w, 0, 0, w, h);
        return bmp;
    }
}
