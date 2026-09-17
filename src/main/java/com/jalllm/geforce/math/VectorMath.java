package com.jalllm.geforce.math;

/**
 * Array and vector operations for high-performance contiguous memory access.
 */
public final class VectorMath {

    private VectorMath() {}

    public static float dot(float[] a, float[] b) {
        if (a.length != b.length) {
            throw new IllegalArgumentException("Vector lengths do not match: " + a.length + " != " + b.length);
        }
        float sum = 0.0f;
        for (int i = 0; i < a.length; i++) {
            sum += a[i] * b[i];
        }
        return sum;
    }

    public static float sum(float[] a) {
        float s = 0.0f;
        for (float v : a) s += v;
        return s;
    }

    public static float mean(float[] a) {
        return sum(a) / a.length;
    }

    public static float max(float[] a) {
        float m = -Float.MAX_VALUE;
        for (float v : a) {
            if (v > m) m = v;
        }
        return m;
    }

    public static float variance(float[] a, float mean) {
        float sumSqDiff = 0.0f;
        for (float v : a) {
            float diff = v - mean;
            sumSqDiff += diff * diff;
        }
        return sumSqDiff / a.length;
    }

    public static float std(float[] a, float mean) {
        return (float) Math.sqrt(variance(a, mean));
    }

    public static void addInPlace(float[] a, float[] b) {
        for (int i = 0; i < a.length; i++) {
            a[i] += b[i];
        }
    }

    public static void scaleInPlace(float[] a, float scale) {
        for (int i = 0; i < a.length; i++) {
            a[i] *= scale;
        }
    }
}