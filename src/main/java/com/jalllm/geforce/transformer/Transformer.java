package com.jalllm.geforce.transformer;

import com.jalllm.geforce.autograd.Variable;
import com.jalllm.geforce.autograd.VariableOps;
import com.jalllm.geforce.normalization.LayerNorm;
import com.jalllm.geforce.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;

/**
 * Decoder-only Transformer.
 * Stacks multiple TransformerBlocks with a final LayerNorm.
 */
public final class Transformer {

    private final List<TransformerBlock> blocks;
    private final LayerNorm finalLn;
    private final int numLayers;
    private final int embedDim;
    private final int numHeads;
    private final int ffDim;

    public Transformer(int numLayers, int embedDim, int numHeads, int ffDim) {
        this.numLayers = numLayers;
        this.embedDim = embedDim;
        this.numHeads = numHeads;
        this.ffDim = ffDim;
        this.blocks = new ArrayList<>();

        for (int i = 0; i < numLayers; i++) {
            blocks.add(new TransformerBlock(embedDim, numHeads, ffDim));
        }

        this.finalLn = new LayerNorm(embedDim);
    }

    public Transformer(int numLayers, int embedDim, int numHeads) {
        this(numLayers, embedDim, numHeads, embedDim * 4);
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
            return Variable.of(forwardInference(x.value(), mask != null ? mask.value() : null), "transformer_out");
        }

        Variable h = x;
        for (TransformerBlock block : blocks) {
            h = block.forward(h, mask);
        }

        // Final LayerNorm
        h = finalLn.forward(h);

        return h;
    }

    private Tensor forwardInference(Tensor x, Tensor mask) {
        Tensor h = x;
        for (TransformerBlock block : blocks) {
            h = block.forward(h, mask);
        }
        h = finalLn.forward(h);
        return h;
    }

    public List<TransformerBlock> blocks() {
        return blocks;
    }

    public LayerNorm finalLn() {
        return finalLn;
    }

    public int numLayers() { return numLayers; }
    public int embedDim() { return embedDim; }
    public int numHeads() { return numHeads; }
    public int ffDim() { return ffDim; }

    public List<Variable> parameters() {
        List<Variable> params = new ArrayList<>();
        for (TransformerBlock block : blocks) {
            params.addAll(block.parameters());
        }
        params.addAll(finalLn.parameters());
        return params;
    }
}