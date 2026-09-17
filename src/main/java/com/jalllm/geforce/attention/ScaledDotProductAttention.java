package com.jalllm.geforce.attention;

import com.jalllm.geforce.autograd.Variable;
import com.jalllm.geforce.autograd.VariableOps;
import com.jalllm.geforce.autograd.ops.SoftmaxOp;
import com.jalllm.geforce.math.MatrixMath;
import com.jalllm.geforce.tensor.Tensor;
import com.jalllm.geforce.tensor.TensorShape;

import java.util.List;

/**
 * Scaled Dot-Product Attention.
 * Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
 */
public final class ScaledDotProductAttention {

    private ScaledDotProductAttention() {}

    /**
     * Computes scaled dot-product attention with autograd support.
     *
     * @param query  Query tensor [batch, heads, seqLen, headDim] or [batch, seqLen, headDim]
     * @param key    Key tensor [batch, heads, seqLen, headDim] or [batch, seqLen, headDim]
     * @param value  Value tensor [batch, heads, seqLen, headDim] or [batch, seqLen, headDim]
     * @param mask   Optional attention mask (broadcastable to attention scores)
     * @return Attention output tensor
     */
    public static Variable forward(Variable query, Variable key, Variable value, Variable mask) {
        return new ScaledDotProductAttentionOp(query, key, value, mask).output();
    }

    /**
     * Computes scaled dot-product attention without autograd (for inference).
     */
    public static Tensor forward(Tensor query, Tensor key, Tensor value, Tensor mask) {
        return forwardInference(query, key, value, mask);
    }

    private static Tensor forwardInference(Tensor query, Tensor key, Tensor value, Tensor mask) {
        // Q @ K^T
        Tensor keyT = MatrixMath.transpose(key);
        Tensor scores = MatrixMath.matmul(query, keyT);

        // Scale by sqrt(d_k)
        int headDim = query.shape()[query.rank() - 1];
        float scale = 1.0f / (float) Math.sqrt(headDim);
        scores = scores.mul(scale);

        // Apply mask if provided
        if (mask != null) {
            scores = scores.add(mask);
        }

        // Softmax
        Tensor attnWeights = com.jalllm.geforce.activation.Softmax.forward(scores);

        // Attention @ V
        Tensor output = MatrixMath.matmul(attnWeights, value);

        return output;
    }

    /**
     * Custom Operation for Scaled Dot-Product Attention with autograd support.
     */
    private static final class ScaledDotProductAttentionOp implements com.jalllm.geforce.autograd.Operation {

        private final Variable query;
        private final Variable key;
        private final Variable value;
        private final Variable mask;
        private final Variable output;
        private final Tensor attnWeights; // Store for backward
        private final float scale;

        ScaledDotProductAttentionOp(Variable query, Variable key, Variable value, Variable mask) {
            this.query = query;
            this.key = key;
            this.value = value;
            this.mask = mask;

            int headDim = query.value().shape()[query.value().rank() - 1];
            this.scale = 1.0f / (float) Math.sqrt(headDim);

            // Forward pass
            Tensor keyT = MatrixMath.transpose(key.value());
            Tensor scores = MatrixMath.matmul(query.value(), keyT);
            scores = scores.mul(scale);

            if (mask != null) {
                scores = scores.add(mask.value());
            }

            this.attnWeights = com.jalllm.geforce.activation.Softmax.forward(scores);
            Tensor outputTensor = MatrixMath.matmul(attnWeights, value.value());
            this.output = Variable.fromOp(outputTensor, this, "attention_out");
        }

        @Override
        public List<Variable> inputs() {
            if (mask != null) {
                return List.of(query, key, value, mask);
            }
            return List.of(query, key, value);
        }

        @Override
        public Variable output() {
            return output;
        }

        @Override
        public void backward(Tensor gradOutput) {
            // Backward pass for attention
            // gradOutput shape: [batch, heads, seqLen, headDim] or [batch, seqLen, headDim]
            // attnWeights shape: [batch, heads, seqLen, seqLen] or [batch, seqLen, seqLen]

            Tensor attnW = attnWeights;
            Tensor v = value.value();
            Tensor q = query.value();
            Tensor k = key.value();

            // grad_V = attnWeights^T @ gradOutput
            Tensor attnWT = MatrixMath.transpose(attnW);
            Tensor gradV = MatrixMath.matmul(attnWT, gradOutput);
            if (value.requiresGrad()) {
                value.accumulateGradient(gradV);
            }

            // grad_attnWeights = gradOutput @ V^T
            Tensor vT = MatrixMath.transpose(v);
            Tensor gradAttnW = MatrixMath.matmul(gradOutput, vT);

            // Backward through softmax
            // Softmax backward: gradScores = attnW * (gradAttnW - sum(gradAttnW * attnW, dim=-1, keepdim=True))
            Tensor gradScores = softmaxBackward(attnW, gradAttnW);

            // Scale gradient
            gradScores = gradScores.mul(scale);

            // grad_Q = gradScores @ K
            Tensor gradQ = MatrixMath.matmul(gradScores, k);
            if (query.requiresGrad()) {
                query.accumulateGradient(gradQ);
            }

            // grad_K = gradScores^T @ Q
            Tensor gradScoresT = MatrixMath.transpose(gradScores);
            Tensor gradK = MatrixMath.matmul(gradScoresT, q);
            if (key.requiresGrad()) {
                key.accumulateGradient(gradK);
            }

            // Mask gradient (if mask requires grad, which it typically doesn't)
            if (mask != null && mask.requiresGrad()) {
                // gradMask = gradScores (since mask is added directly)
                mask.accumulateGradient(gradScores);
            }
        }

        private Tensor softmaxBackward(Tensor softmaxOut, Tensor gradOut) {
            // gradIn_i = y_i * (gradOut_i - sum_j(gradOut_j * y_j))
            int[] shape = softmaxOut.shape();
            int rank = shape.length;
            int lastDim = shape[rank - 1];
            int outerSize = 1;
            for (int i = 0; i < rank - 1; i++) {
                outerSize *= shape[i];
            }

            float[] y = softmaxOut.data();
            float[] gradOutData = gradOut.data();
            float[] gradIn = new float[y.length];

            for (int b = 0; b < outerSize; b++) {
                int offset = b * lastDim;

                float dot = 0.0f;
                for (int i = 0; i < lastDim; i++) {
                    dot += gradOutData[offset + i] * y[offset + i];
                }

                for (int i = 0; i < lastDim; i++) {
                    gradIn[offset + i] = y[offset + i] * (gradOutData[offset + i] - dot);
                }
            }

            return new Tensor(shape, gradIn);
        }

        @Override
        public String name() {
            return "ScaledDotProductAttention";
        }
    }
}
