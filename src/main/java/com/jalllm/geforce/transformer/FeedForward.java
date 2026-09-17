package com.jalllm.geforce.transformer;

import com.jalllm.geforce.autograd.Variable;
import com.jalllm.geforce.autograd.VariableOps;
import com.jalllm.geforce.activation.GELU;
import com.jalllm.geforce.math.MatrixMath;
import com.jalllm.geforce.math.RandomInitializer;
import com.jalllm.geforce.tensor.Tensor;

/**
 * Position-wise Feed-Forward Network.
 * FFN(x) = GELU(x @ W1 + b1) @ W2 + b2
 * Typically: embedDim -> 4*embedDim -> embedDim
 */
public final class FeedForward {

    private final Variable w1;
    private final Variable b1;
    private final Variable w2;
    private final Variable b2;
    private final int embedDim;
    private final int ffDim;

    public FeedForward(int embedDim, int ffDim) {
        this.embedDim = embedDim;
        this.ffDim = ffDim;

        // First projection: embedDim -> ffDim
        this.w1 = Variable.of(new Tensor(new int[]{embedDim, ffDim}), "ffn_w1");
        this.b1 = Variable.of(new Tensor(new int[]{ffDim}), "ffn_b1");

        // Second projection: ffDim -> embedDim
        this.w2 = Variable.of(new Tensor(new int[]{ffDim, embedDim}), "ffn_w2");
        this.b2 = Variable.of(new Tensor(new int[]{embedDim}), "ffn_b2");

        // Initialize
        RandomInitializer.xavierUniform(w1.value(), embedDim, ffDim);
        RandomInitializer.xavierUniform(w2.value(), ffDim, embedDim);
        RandomInitializer.constant(b1.value(), 0.0f);
        RandomInitializer.constant(b2.value(), 0.0f);
    }

    public FeedForward(int embedDim) {
        this(embedDim, embedDim * 4);
    }

    /**
     * Forward pass with autograd support.
     * Input: [batch, seqLen, embedDim]
     * Output: [batch, seqLen, embedDim]
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
            return Variable.of(forwardInference(x.value()), "ffn_out");
        }

        // x @ W1 + b1
        Variable hidden = VariableOps.add(VariableOps.matmul(x, w1), b1);

        // GELU activation
        Variable activated = VariableOps.gelu(hidden);

        // activated @ W2 + b2
        Variable out = VariableOps.add(VariableOps.matmul(activated, w2), b2);

        return out;
    }

    private Tensor forwardInference(Tensor x) {
        // x @ W1 + b1
        Tensor hidden = MatrixMath.matmul(x, w1.value()).add(b1.value());

        // GELU
        Tensor activated = GELU.forward(hidden);

        // activated @ W2 + b2
        Tensor out = MatrixMath.matmul(activated, w2.value()).add(b2.value());

        return out;
    }

    // Getters for parameters
    public Variable w1() { return w1; }
    public Variable b1() { return b1; }
    public Variable w2() { return w2; }
    public Variable b2() { return b2; }

    public int embedDim() { return embedDim; }
    public int ffDim() { return ffDim; }

    public java.util.List<Variable> parameters() {
        return java.util.List.of(w1, b1, w2, b2);
    }
}