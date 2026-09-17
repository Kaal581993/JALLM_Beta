package com.jalllm.geforce.activation;

import com.jalllm.geforce.math.MathUtils;
import com.jalllm.geforce.tensor.Tensor;

/**
 * SiLU (Sigmoid Linear Unit / Swish) Activation function.
 */
public final class SiLU {

    public static Tensor forward(Tensor input) {
        float[] inData = input.data();
        float[] outData = new float[inData.length];
        for (int i = 0; i < inData.length; i++) {
            outData[i] = MathUtils.silu(inData[i]);
        }
        return new Tensor(input.shape(), outData);
    }

    /**
     * Backward pass for SiLU.
     * dy/dx = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
     */
    public static Tensor backward(Tensor input, Tensor gradOutput) {
        float[] inData = input.data();
        float[] gradOut = gradOutput.data();
        float[] gradIn = new float[inData.length];

        for (int i = 0; i < inData.length; i++) {
            float x = inData[i];
            float sig = MathUtils.sigmoid(x);
            float dy_dx = sig * (1.0f + x * (1.0f - sig));
            gradIn[i] = gradOut[i] * dy_dx;
        }

        return new Tensor(input.shape(), gradIn);
    }
}