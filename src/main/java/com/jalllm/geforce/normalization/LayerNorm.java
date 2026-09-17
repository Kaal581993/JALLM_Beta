package com.jalllm.geforce.normalization;

import com.jalllm.geforce.autograd.Variable;
import com.jalllm.geforce.autograd.VariableOps;
import com.jalllm.geforce.math.MathUtils;
import com.jalllm.geforce.math.RandomInitializer;
import com.jalllm.geforce.tensor.Tensor;
import com.jalllm.geforce.tensor.TensorShape;

import java.util.Arrays;

/**
 * Layer Normalization.
 * Normalizes over the last dimension of the input tensor.
 * y = (x - mean) / sqrt(var + eps) * weight + bias
 */
public final class LayerNorm {

    private final Variable weight;
    private final Variable bias;
    private final float eps;

    public LayerNorm(int normalizedShape, float eps) {
        this.eps = eps;
        this.weight = Variable.of(Tensor.ones(normalizedShape), "ln_weight");
        this.bias = Variable.of(Tensor.zeros(normalizedShape), "ln_bias");
        RandomInitializer.constant(weight.value(), 1.0f);
        RandomInitializer.constant(bias.value(), 0.0f);
    }

    public LayerNorm(int normalizedShape) {
        this(normalizedShape, 1e-5f);
    }

    /**
     * Forward pass with autograd support.
     */
    public Variable forward(Variable x) {
        return forward(x, true);
    }

    /**
     * Forward pass without autograd (for inference).
     */
    public Tensor forward(Tensor x) {
        return forwardInference(x);
    }

    private Variable forward(Variable x, boolean trackGrad) {
        if (!trackGrad) {
            return Variable.of(forwardInference(x.value()), "ln_out");
        }

        // Compute mean and variance along last dimension
        int[] shape = x.value().shape();
        int rank = shape.length;
        int lastDim = shape[rank - 1];
        int outerSize = 1;
        for (int i = 0; i < rank - 1; i++) {
            outerSize *= shape[i];
        }

        // We need to implement LayerNorm as a custom operation for autograd
        return new LayerNormOp(x, weight, bias, eps, outerSize, lastDim).output();
    }

    private Tensor forwardInference(Tensor x) {
        int[] shape = x.shape();
        int rank = shape.length;
        int lastDim = shape[rank - 1];
        int outerSize = 1;
        for (int i = 0; i < rank - 1; i++) {
            outerSize *= shape[i];
        }

        float[] inData = x.data();
        float[] outData = new float[inData.length];
        float[] wData = weight.value().data();
        float[] bData = bias.value().data();

        for (int b = 0; b < outerSize; b++) {
            int offset = b * lastDim;

            // Compute mean
            float mean = 0.0f;
            for (int i = 0; i < lastDim; i++) {
                mean += inData[offset + i];
            }
            mean /= lastDim;

            // Compute variance
            float var = 0.0f;
            for (int i = 0; i < lastDim; i++) {
                float diff = inData[offset + i] - mean;
                var += diff * diff;
            }
            var /= lastDim;

            // Normalize and scale
            float invStd = (float) (1.0 / Math.sqrt(var + eps));
            for (int i = 0; i < lastDim; i++) {
                float normalized = (inData[offset + i] - mean) * invStd;
                outData[offset + i] = normalized * wData[i] + bData[i];
            }
        }

        return new Tensor(shape, outData);
    }

    public Variable weight() {
        return weight;
    }

    public Variable bias() {
        return bias;
    }

    public float eps() {
        return eps;
    }

    /**
     * Returns all parameters (weight and bias) for optimization.
     */
    public java.util.List<Variable> parameters() {
        return java.util.List.of(weight, bias);
    }

    /**
     * Custom Operation for LayerNorm with autograd support.
     */
    private static final class LayerNormOp implements com.jalllm.geforce.autograd.Operation {

        private final Variable x;
        private final Variable weight;
        private final Variable bias;
        private final float eps;
        private final int outerSize;
        private final int lastDim;
        private final Variable output;

        LayerNormOp(Variable x, Variable weight, Variable bias, float eps, int outerSize, int lastDim) {
            this.x = x;
            this.weight = weight;
            this.bias = bias;
            this.eps = eps;
            this.outerSize = outerSize;
            this.lastDim = lastDim;

            Tensor result = forwardInference(x.value(), weight.value(), bias.value(), eps, outerSize, lastDim);
            this.output = com.jalllm.geforce.autograd.Variable.fromOp(result, this, "layernorm");
        }

        @Override
        public java.util.List<Variable> inputs() {
            return java.util.List.of(x, weight, bias);
        }

        @Override
        public Variable output() {
            return output;
        }

        @Override
        public void backward(Tensor gradOutput) {
            // LayerNorm backward pass
            // Reference: https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/LayerNorm.cpp
            float[] xData = x.value().data();
            float[] wData = weight.value().data();
            float[] bData = bias.value().data();
            float[] gradOutData = gradOutput.data();

            float[] gradXData = new float[xData.length];
            float[] gradWData = new float[wData.length];
            float[] gradBData = new float[bData.length];

            for (int b = 0; b < outerSize; b++) {
                int offset = b * lastDim;

                // Recompute mean and variance
                float mean = 0.0f;
                for (int i = 0; i < lastDim; i++) {
                    mean += xData[offset + i];
                }
                mean /= lastDim;

                float var = 0.0f;
                for (int i = 0; i < lastDim; i++) {
                    float diff = xData[offset + i] - mean;
                    var += diff * diff;
                }
                var /= lastDim;

                float invStd = (float) (1.0 / Math.sqrt(var + eps));

                // Compute gradients for weight and bias
                for (int i = 0; i < lastDim; i++) {
                    float normalized = (xData[offset + i] - mean) * invStd;
                    gradWData[i] += gradOutData[offset + i] * normalized;
                    gradBData[i] += gradOutData[offset + i];
                }

                // Compute gradient for input
                // dx = (1/N) * w * invStd * (N * dy - sum(dy) - normalized * sum(dy * normalized))
                float sumDy = 0.0f;
                float sumDyNorm = 0.0f;
                for (int i = 0; i < lastDim; i++) {
                    float normalized = (xData[offset + i] - mean) * invStd;
                    sumDy += gradOutData[offset + i];
                    sumDyNorm += gradOutData[offset + i] * normalized;
                }

                for (int i = 0; i < lastDim; i++) {
                    float normalized = (xData[offset + i] - mean) * invStd;
                    float dy = gradOutData[offset + i];
                    float dx = wData[i] * invStd * (dy - sumDy / lastDim - normalized * sumDyNorm / lastDim);
                    gradXData[offset + i] = dx;
                }
            }

            if (x.requiresGrad()) {
                x.accumulateGradient(new Tensor(x.value().shape(), gradXData));
            }
            if (weight.requiresGrad()) {
                weight.accumulateGradient(new Tensor(weight.value().shape(), gradWData));
            }
            if (bias.requiresGrad()) {
                bias.accumulateGradient(new Tensor(bias.value().shape(), gradBData));
            }
        }

        @Override
        public String name() {
            return "LayerNorm";
        }

        private static Tensor forwardInference(Tensor x, Tensor weight, Tensor bias, float eps, int outerSize, int lastDim) {
            float[] inData = x.data();
            float[] outData = new float[inData.length];
            float[] wData = weight.data();
            float[] bData = bias.data();

            for (int b = 0; b < outerSize; b++) {
                int offset = b * lastDim;

                float mean = 0.0f;
                for (int i = 0; i < lastDim; i++) {
                    mean += inData[offset + i];
                }
                mean /= lastDim;

                float var = 0.0f;
                for (int i = 0; i < lastDim; i++) {
                    float diff = inData[offset + i] - mean;
                    var += diff * diff;
                }
                var /= lastDim;

                float invStd = (float) (1.0 / Math.sqrt(var + eps));
                for (int i = 0; i < lastDim; i++) {
                    float normalized = (inData[offset + i] - mean) * invStd;
                    outData[offset + i] = normalized * wData[i] + bData[i];
                }
            }

            return new Tensor(x.shape(), outData);
        }
    }
}