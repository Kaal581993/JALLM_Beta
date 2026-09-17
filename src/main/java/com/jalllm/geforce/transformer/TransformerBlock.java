package com.jalllm.geforce.transformer;

import com.jalllm.geforce.attention.MultiHeadAttention;
import com.jalllm.geforce.autograd.Variable;
import com.jalllm.geforce.autograd.VariableOps;
import com.jalllm.geforce.normalization.LayerNorm;
import com.jalllm.geforce.tensor.Tensor;

/**
 * Transformer Block (Decoder-only).
 * Architecture:
 *   x -> LayerNorm -> MultiHeadAttention -> Residual -> LayerNorm -> FeedForward -> Residual -> out
 */
public final class TransformerBlock {

    private final LayerNorm ln1;
    private final MultiHeadAttention attention;
    private final LayerNorm ln2;
    private final FeedForward ffn;
    private final int embedDim;

    public TransformerBlock(int embedDim, int numHeads, int ffDim) {
        this.embedDim = embedDim;
        this.ln1 = new LayerNorm(embedDim);
        this.attention = new MultiHeadAttention(embedDim, numHeads);
        this.ln2 = new LayerNorm(embedDim);
        this.ffn = new FeedForward(embedDim, ffDim);
    }

    public TransformerBlock(int embedDim, int numHeads) {
        this(embedDim, numHeads, embedDim * 4);
    }

    /**
     * Forward pass with autograd support.
     * Input: [batch, seqLen, embedDim]
     * Output: [batch, seqLen, embedDim]
     */
    public Variable forward(Variable x, Variable mask) {
        return forward(x, mask, true);
    }

    /**
     * Forward pass without autograd (for inference).
     */
    public Tensor forward(Tensor x, Tensor mask) {
        return forwardInference(x, mask);
    }

    private Variable forward(Variable x, Variable mask, boolean trackGrad) {
        if (!trackGrad) {
            return Variable.of(forwardInference(x.value(), mask != null ? mask.value() : null), "block_out");
        }

        // Pre-LN architecture (more stable)
        // x -> LN1 -> Attention -> Residual -> LN2 -> FFN -> Residual

        // First sub-layer: Attention with residual
        Variable ln1Out = ln1.forward(x);
        Variable attnOut = attention.forward(ln1Out, mask);
        Variable residual1 = VariableOps.add(x, attnOut);

        // Second sub-layer: FFN with residual
        Variable ln2Out = ln2.forward(residual1);
        Variable ffnOut = ffn.forward(ln2Out);
        Variable residual2 = VariableOps.add(residual1, ffnOut);

        return residual2;
    }

    private Tensor forwardInference(Tensor x, Tensor mask) {
        // First sub-layer
        Tensor ln1Out = ln1.forward(x);
        Tensor attnOut = attention.forward(ln1Out, mask);
        Tensor residual1 = attnOut.add(x);

        // Second sub-layer
        Tensor ln2Out = ln2.forward(residual1);
        Tensor ffnOut = ffn.forward(ln2Out);
        Tensor residual2 = residual1.add(ffnOut);

        return residual2;
    }

    // Getters for submodules
    public LayerNorm ln1() { return ln1; }
    public MultiHeadAttention attention() { return attention; }
    public LayerNorm ln2() { return ln2; }
    public FeedForward ffn() { return ffn; }

    public int embedDim() { return embedDim; }

    public java.util.List<Variable> parameters() {
        java.util.List<Variable> params = new java.util.ArrayList<>();
        params.addAll(ln1.parameters());
        params.addAll(attention.parameters());
        params.addAll(ln2.parameters());
        params.addAll(ffn.parameters());
        return params;
    }
}