package com.jalllm.geforce.activation;

import com.jalllm.geforce.tensor.Tensor;
import com.jalllm.geforce.tensor.TensorShape;

/**
 * Stable Softmax activation function.
 * Computes softmax along the last dimension of any N-dimensional tensor.
 */
public final class Softmax {

    /**
     * Forward pass of Softmax over the last dimension.
     */
    public static Tensor forward(Tensor input) {
        int[] shape = input.shape();
        int rank = shape.length;
        if (rank == 0) {
            throw new IllegalArgumentException("Cannot compute softmax on a scalar (rank 0) tensor.");
        }

        int lastDim = shape[rank - 1];
        int outerSize = 1;
        for (int i = 0; i < rank - 1; i++) {
            outerSize *= shape[i];
        }

        float[] inData = input.data();
        float[] outData = new float[inData.length];

        for (int b = 0; b < outerSize; b++) {
            int offset = b * lastDim;

            // 1. Find max for stability
            float max = -Float.MAX_VALUE;
            for (int i = 0; i < lastDim; i++) {
                float val = inData[offset + i];
                if (val > max) max = val;
            }

            // 2. Compute sum of exponentials
            float sum = 0.0f;
            for (int i = 0; i < lastDim; i++) {
                float exp = (float) Math.exp(inData[offset + i] - max);
                outData[offset + i] = exp;
                sum += exp;
            }

            // 3. Divide by sum
            for (int i = 0; i < lastDim; i++) {
                outData[offset + i] /= sum;
            }
        }

        return new Tensor(shape, outData);
    }

    /**
     * Backward pass for Softmax.
     * gradIn_i = y_i * (gradOut_i - sum_j(gradOut_j * y_j))
     * where y is the forward softmax output tensor.
     */
    public static Tensor backward(Tensor softmaxOutput, Tensor gradOutput) {
        int[] shape = softmaxOutput.shape();
        if (!TensorShape.compatible(shape, gradOutput.shape())) {
            throw new IllegalArgumentException("Softmax output and gradient must have identical shapes");
        }
        int rank = shape.length;
        int lastDim = shape[rank - 1];
        int outerSize = 1;
        for (int i = 0; i < rank - 1; i++) {
            outerSize *= shape[i];
        }

        float[] y = softmaxOutput.data();
        float[] gradOut = gradOutput.data();
        float[] gradIn = new float[y.length];

        for (int b = 0; b < outerSize; b++) {
            int offset = b * lastDim;

            // Compute dot product of gradOut and y for this row
            float dot = 0.0f;
            for (int i = 0; i < lastDim; i++) {
                dot += gradOut[offset + i] * y[offset + i];
            }

            // gradIn = y * (gradOut - dot)
            for (int i = 0; i < lastDim; i++) {
                gradIn[offset + i] = y[offset + i] * (gradOut[offset + i] - dot);
            }
        }

        return new Tensor(shape, gradIn);
    }
}
