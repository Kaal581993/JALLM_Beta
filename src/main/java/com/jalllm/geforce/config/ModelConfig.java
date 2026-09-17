package com.jalllm.geforce.config;

import java.io.Serializable;

/**
 * Model architecture configuration.
 */
public final class ModelConfig implements Serializable {

    private static final long serialVersionUID = 1L;

    private String modelType = "MiniGPT";
    private int vocabSize = 8000;
    private int embedDim = 256;
    private int maxSeqLen = 128;
    private int numLayers = 4;
    private int numHeads = 4;
    private int ffDim = 1024; // 4 * embedDim
    private float dropout = 0.1f;
    private boolean weightTying = true;
    private String activation = "gelu"; // gelu, silu, relu

    public ModelConfig() {}

    public ModelConfig(int vocabSize, int embedDim, int maxSeqLen, int numLayers, int numHeads) {
        this.vocabSize = vocabSize;
        this.embedDim = embedDim;
        this.maxSeqLen = maxSeqLen;
        this.numLayers = numLayers;
        this.numHeads = numHeads;
        this.ffDim = embedDim * 4;
    }

    public ModelConfig(int embedDim, int maxSeqLen, int numLayers, int numHeads, int ffDim, int vocabSize, int headDim, float dropout) {
        this.embedDim = embedDim;
        this.maxSeqLen = maxSeqLen;
        this.numLayers = numLayers;
        this.numHeads = numHeads;
        this.ffDim = ffDim;
        this.vocabSize = vocabSize;
        this.dropout = dropout;
    }

    // Getters and setters
    public String modelType() { return modelType; }
    public void setModelType(String modelType) { this.modelType = modelType; }

    public int vocabSize() { return vocabSize; }
    public void setVocabSize(int vocabSize) { this.vocabSize = vocabSize; }

    public int embedDim() { return embedDim; }
    public void setEmbedDim(int embedDim) { this.embedDim = embedDim; }

    public int maxSeqLen() { return maxSeqLen; }
    public void setMaxSeqLen(int maxSeqLen) { this.maxSeqLen = maxSeqLen; }

    public int numLayers() { return numLayers; }
    public void setNumLayers(int numLayers) { this.numLayers = numLayers; }

    public int numHeads() { return numHeads; }
    public void setNumHeads(int numHeads) { this.numHeads = numHeads; }

    public int ffDim() { return ffDim; }
    public void setFfDim(int ffDim) { this.ffDim = ffDim; }

    public float dropout() { return dropout; }
    public void setDropout(float dropout) { this.dropout = dropout; }

    public boolean weightTying() { return weightTying; }
    public void setWeightTying(boolean weightTying) { this.weightTying = weightTying; }

    public String activation() { return activation; }
    public void setActivation(String activation) { this.activation = activation; }

    public int headDim() {
        return embedDim / numHeads;
    }

    public long estimatedParamCount() {
        // Embedding: vocabSize * embedDim + maxSeqLen * embedDim
        long embParams = (long) vocabSize * embedDim + (long) maxSeqLen * embedDim;

        // Transformer layers
        long layerParams = 0;
        for (int i = 0; i < numLayers; i++) {
            // LayerNorm 1: 2 * embedDim
            // Attention: 4 * embedDim * embedDim (Q, K, V, O) + 4 * embedDim (biases)
            // LayerNorm 2: 2 * embedDim
            // FFN: 2 * embedDim * ffDim + embedDim + ffDim (biases)
            layerParams += 2 * embedDim; // ln1
            layerParams += 4L * embedDim * embedDim + 4 * embedDim; // attention
            layerParams += 2 * embedDim; // ln2
            layerParams += 2L * embedDim * ffDim + embedDim + ffDim; // ffn
        }

        // Final LayerNorm: 2 * embedDim
        // LM Head bias: vocabSize (weight tied)
        long finalParams = 2 * embedDim + vocabSize;

        return embParams + layerParams + finalParams;
    }

    @Override
    public String toString() {
        return String.format("ModelConfig(type=%s, vocab=%d, dim=%d, seqLen=%d, layers=%d, heads=%d, ffDim=%d, params≈%d)",
                modelType, vocabSize, embedDim, maxSeqLen, numLayers, numHeads, ffDim, estimatedParamCount());
    }
}