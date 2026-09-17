package com.jalllm.geforce.optimizer;

import com.jalllm.geforce.autograd.Variable;
import com.jalllm.geforce.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;

/**
 * AdamW optimizer.
 * Adam with decoupled weight decay regularization.
 * Reference: "Fixing Weight Decay Regularization in Adam" (Loshchilov & Hutter, 2019)
 */
public final class AdamW implements Optimizer {

    private final List<Variable> parameters;
    private final float lr;
    private final float beta1;
    private final float beta2;
    private final float eps;
    private final float weightDecay;
    private final float[] m; // First moment estimates
    private final float[] v; // Second moment estimates
    private int stepCount;

    public AdamW(List<Variable> parameters, float lr, float beta1, float beta2, float eps, float weightDecay) {
        this.parameters = new ArrayList<>(parameters);
        this.lr = lr;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.eps = eps;
        this.weightDecay = weightDecay;
        this.stepCount = 0;

        // Initialize moment buffers
        int totalParams = 0;
        for (Variable p : parameters) {
            totalParams += p.value().elementCount();
        }
        this.m = new float[totalParams];
        this.v = new float[totalParams];
    }

    public AdamW(List<Variable> parameters, float lr) {
        this(parameters, lr, 0.9f, 0.999f, 1e-8f, 0.01f);
    }

    public AdamW(List<Variable> parameters, float lr, float weightDecay) {
        this(parameters, lr, 0.9f, 0.999f, 1e-8f, weightDecay);
    }

    public AdamW(List<Variable> parameters) {
        this(parameters, 3e-4f);
    }

    @Override
    public void step() {
        stepCount++;
        float beta1Pow = (float) Math.pow(beta1, stepCount);
        float beta2Pow = (float) Math.pow(beta2, stepCount);

        int offset = 0;
        for (Variable p : parameters) {
            if (!p.requiresGrad() || p.gradient() == null) {
                offset += p.value().elementCount();
                continue;
            }

            float[] paramData = p.value().data();
            float[] gradData = p.gradient().data();
            int paramSize = paramData.length;

            for (int i = 0; i < paramSize; i++) {
                float grad = gradData[i];

                // Weight decay (decoupled)
                if (weightDecay != 0.0f) {
                    paramData[i] -= lr * weightDecay * paramData[i];
                }

                // Update moments
                m[offset + i] = beta1 * m[offset + i] + (1.0f - beta1) * grad;
                v[offset + i] = beta2 * v[offset + i] + (1.0f - beta2) * grad * grad;

                // Bias correction
                float mHat = m[offset + i] / (1.0f - (float) Math.pow(beta1, stepCount));
                float vHat = v[offset + i] / (1.0f - (float) Math.pow(beta2, stepCount));

                // Update parameter
                paramData[i] -= lr * mHat / ((float) Math.sqrt(vHat) + eps);
            }

            offset += paramSize;
        }
    }

    @Override
    public void zeroGrad() {
        for (Variable p : parameters) {
            p.zeroGrad();
        }
    }

    @Override
    public float learningRate() {
        return lr;
    }

    @Override
    public void setLearningRate(float lr) {
        // Not supported in this simple implementation
        throw new UnsupportedOperationException("Learning rate scheduling not implemented");
    }

    @Override
    public List<Variable> parameters() {
        return new ArrayList<>(parameters);
    }

    public int stepCount() {
        return stepCount;
    }

    public float beta1() { return beta1; }
    public float beta2() { return beta2; }
    public float eps() { return eps; }
    public float weightDecay() { return weightDecay; }
}
