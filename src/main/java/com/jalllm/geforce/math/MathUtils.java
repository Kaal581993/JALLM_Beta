package com.jalllm.geforce.math;

/**
 * Utility mathematical routines for activations and normalizations.
 */
public final class MathUtils {

    private MathUtils() {}

    public static float sigmoid(float x) {
        if (x >= 0) {
            return (float) (1.0 / (1.0 + Math.exp(-x)));
        } else {
            float exp = (float) Math.exp(x);
            return exp / (1.0f + exp);
        }
    }

    /**
     * GELU activation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
     */
    public static float gelu(float x) {
        double constant = Math.sqrt(2.0 / Math.PI);
        double val = constant * (x + 0.044715 * Math.pow(x, 3));
        return (float) (0.5 * x * (1.0 + Math.tanh(val)));
    }

    /**
     * SiLU (Swish) activation: x * sigmoid(x)
     */
    public static float silu(float x) {
        return x * sigmoid(x);
    }
}