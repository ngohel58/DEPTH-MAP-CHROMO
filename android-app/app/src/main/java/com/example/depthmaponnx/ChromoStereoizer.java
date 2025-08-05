package com.example.depthmaponnx;

import android.graphics.Bitmap;
import android.graphics.Color;

import org.opencv.android.OpenCVLoader;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

/**
 * Applies a chromostereopsis effect based on a depth map.
 */
public class ChromoStereoizer {

    static {
        OpenCVLoader.initDebug();
    }

    private static float[][] preprocessDepth(float[][] depth, float smoothingRadius) {
        if (smoothingRadius <= 0) return depth;
        int h = depth.length;
        int w = depth[0].length;
        Mat mat = new Mat(h, w, CvType.CV_32F);
        for (int y = 0; y < h; y++) {
            mat.put(y, 0, depth[y]);
        }
        Mat u8 = new Mat();
        mat.convertTo(u8, CvType.CV_8U, 255.0);
        Mat filtered = new Mat();
        double sigma = Math.max(smoothingRadius * 10.0, 1.0);
        Imgproc.bilateralFilter(u8, filtered, 5, sigma, sigma);
        Mat back = new Mat();
        filtered.convertTo(back, CvType.CV_32F, 1.0 / 255.0);
        float[][] out = new float[h][w];
        for (int y = 0; y < h; y++) {
            back.get(y, 0, out[y]);
        }
        return out;
    }

    public static Bitmap applyEffect(Bitmap original, float[][] depth,
                                     int threshold, int depthScale, int feather,
                                     int redBrightness, int blueBrightness,
                                     int gamma, int blackLevel, int whiteLevel,
                                     int smoothing) {
        int width = original.getWidth();
        int height = original.getHeight();

        float black = blackLevel * 2.55f;
        float white = whiteLevel * 2.55f;
        int[] grayPixels = new int[width * height];
        original.getPixels(grayPixels, 0, width, 0, 0, width, height);
        float[] gray = new float[width * height];
        for (int i = 0; i < grayPixels.length; i++) {
            int c = grayPixels[i];
            float g = 0.299f * Color.red(c) + 0.587f * Color.green(c) + 0.114f * Color.blue(c);
            g = (g - black) / Math.max(white - black, 1e-6f);
            g = Math.max(0f, Math.min(1f, g));
            float gammaVal = (float) (0.1 + (gamma / 100.0) * 2.9);
            gray[i] = (float) Math.pow(g, gammaVal);
        }

        float[][] depthProcessed = preprocessDepth(depth, smoothing / 10.0f);

        double thresholdNorm = threshold / 100.0;
        double steep = Math.max(depthScale, 1e-3);
        double featherNorm = feather / 100.0;
        double steepAdj = steep / (featherNorm * 10.0 + 1.0);

        float redFactor = redBrightness / 50f;
        float blueFactor = blueBrightness / 50f;

        Bitmap output = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        int[] outPixels = new int[width * height];
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                float d = depthProcessed[y][x];
                double blend = 1.0 / (1.0 + Math.exp(-steepAdj * (d - thresholdNorm)));
                float g = gray[y * width + x];
                int r = (int) Math.max(0, Math.min(255, redFactor * g * (float) blend * 255f));
                int b = (int) Math.max(0, Math.min(255, blueFactor * g * (float) (1.0 - blend) * 255f));
                outPixels[y * width + x] = Color.rgb(r, 0, b);
            }
        }
        output.setPixels(outPixels, 0, width, 0, 0, width, height);
        return output;
    }
}
