package com.jalllm.geforce.math;

import com.jalllm.geforce.tensor.Tensor;
import java.util.Random;

/**
 * Implements common deep learning weight initialization methods.
 */
public final class RandomInitializer {

    private static final Random random = new Random();

    private RandomInitializer() {}

    /**
     * Initializes a tensor with Kaiming (He) normal initialization.
     * Ideal for ReLU/GELU activations.
     * σ = sqrt(2 / fanIn)
     */
    public static void kaimingNormal(Tensor tensor, int fanIn) {
        float std = (float) Math.sqrt(2.0 / fanIn);
        float[] d = tensor.data();
        for (int i = 0; i < d.length; i++) {
            d[i] = (float) random.nextGaussian() * std;
        }
    }

    /**
     * Initializes a tensor with Glorot (Xavier) uniform initialization.
     * Ideal for Tanh/Sigmoid activations.
     * limit = sqrt(6 / (fanIn + fanOut))
     */
    public static void xavierUniform(Tensor tensor, int fanIn, int fanOut) {
        float limit = (float) Math.sqrt(6.0 / (fanIn + fanOut));
        float[] d = tensor.data();
        for (int i = 0; i < d.length; i++) {
            d[i] = -limit + random.nextFloat() * (2.0f * limit);
        }
    }

    /**
     * Initializes a tensor with Glorot (Xavier) normal initialization.
     * σ = sqrt(2 / (fanIn + fanOut))
     */
    public static void xavierNormal(Tensor tensor, int fanIn, int fanOut) {
        float std = (float) Math.sqrt(2.0 / (fanIn + fanOut));
        float[] d = tensor.data();
        for (int i = 0; i < d.length; i++) {
            d[i] = (float) random.nextGaussian() * std;
        }
    }

    /**
     * Fills a tensor with constant value.
     */
    public static void constant(Tensor tensor, float val) {
        tensor.fill(val);
    }
}