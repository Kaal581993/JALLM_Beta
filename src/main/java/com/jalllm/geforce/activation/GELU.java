package com.jalllm.geforce.activation;

import com.jalllm.geforce.math.MathUtils;
import com.jalllm.geforce.tensor.Tensor;

/**
 * GELU (Gaussian Error Linear Unit) Activation function.
 */
public final class GELU {

    public static Tensor forward(Tensor input) {
        float[] inData = input.data();
        float[] outData = new float[inData.length];
        for (int i = 0; i < inData.length; i++) {
            outData[i] = MathUtils.gelu(inData[i]);
        }
        return new Tensor(input.shape(), outData);
    }

    /**
     * Backward pass for GELU.
     * dL/dx = dL/dy * dy/dx
     */
    public static Tensor backward(Tensor input, Tensor gradOutput) {
        float[] inData = input.data();
        float[] gradOut = gradOutput.data();
        float[] gradIn = new float[inData.length];

        double constSqrt = Math.sqrt(2.0 / Math.PI);

        for (int i = 0; i < inData.length; i++) {
            float x = inData[i];
            double x3 = x * x * x;
            double inner = constSqrt * (x + 0.044715 * x3);
            double tanhInner = Math.tanh(inner);
            double sech2Inner = 1.0 - tanhInner * tanhInner;
            double dInner = constSqrt * (1.0 + 3 * 0.044715 * x * x);
            
            double dy_dx = 0.5 * (1.0 + tanhInner) + 0.5 * x * sech2Inner * dInner;
            gradIn[i] = (float) (gradOut[i] * dy_dx);
        }

        return new Tensor(input.shape(), gradIn);
    }
}