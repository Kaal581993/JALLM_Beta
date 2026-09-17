package com.jalllm.geforce.embedding;

import com.jalllm.geforce.autograd.Variable;
import com.jalllm.geforce.math.RandomInitializer;
import com.jalllm.geforce.tensor.Tensor;
import com.jalllm.geforce.tensor.TensorShape;

import java.util.Arrays;

/**
 * Token Embedding layer with optional positional embeddings.
 * Maps token IDs to dense vectors.
 */
public final class TokenEmbedding {

    private final Variable tokenEmbedding;
    private final Variable positionEmbedding;
    private final int vocabSize;
    private final int embedDim;
    private final int maxSeqLen;
    private final float dropout; // Not implemented yet, placeholder

    public TokenEmbedding(int vocabSize, int embedDim, int maxSeqLen) {
        this.vocabSize = vocabSize;
        this.embedDim = embedDim;
        this.maxSeqLen = maxSeqLen;
        this.dropout = 0.0f;

        // Token embeddings: [vocabSize, embedDim]
        this.tokenEmbedding = Variable.of(new Tensor(new int[]{vocabSize, embedDim}), "token_emb");

        // Position embeddings: [maxSeqLen, embedDim]
        this.positionEmbedding = Variable.of(new Tensor(new int[]{maxSeqLen, embedDim}), "pos_emb");

        // Initialize
        RandomInitializer.xavierUniform(tokenEmbedding.value(), vocabSize, embedDim);
        RandomInitializer.xavierUniform(positionEmbedding.value(), maxSeqLen, embedDim);
    }

    public TokenEmbedding(int vocabSize, int embedDim) {
        this(vocabSize, embedDim, 2048);
    }

    /**
     * Forward pass with autograd support.
     * Input: token IDs tensor [batch, seqLen]
     * Output: embeddings [batch, seqLen, embedDim]
     */
    public Variable forward(Variable tokenIds) {
        return forward(tokenIds, true);
    }

    /**
     * Forward pass without autograd (for inference).
     */
    public Tensor forward(Tensor tokenIds) {
        return forwardInference(tokenIds);
    }

    private Variable forward(Variable tokenIds, boolean trackGrad) {
        if (!trackGrad) {
            return Variable.of(forwardInference(tokenIds.value()), "emb_out");
        }

        // This is a custom operation for embedding lookup
        return new EmbeddingOp(tokenIds, tokenEmbedding, positionEmbedding).output();
    }

    private Tensor forwardInference(Tensor tokenIds) {
        int[] shape = tokenIds.shape();
        int batch = shape[0];
        int seqLen = shape[1];

        if (seqLen > maxSeqLen) {
            throw new IllegalArgumentException("Sequence length " + seqLen + " exceeds maxSeqLen " + maxSeqLen);
        }

        float[] tokenIdsData = tokenIds.data();
        float[] tokenEmbData = tokenEmbedding.value().data();
        float[] posEmbData = positionEmbedding.value().data();

        float[] outData = new float[batch * seqLen * embedDim];

        for (int b = 0; b < batch; b++) {
            for (int s = 0; s < seqLen; s++) {
                int tokenId = (int) tokenIdsData[b * seqLen + s];
                if (tokenId < 0 || tokenId >= vocabSize) {
                    throw new IllegalArgumentException("Token ID " + tokenId + " out of range [0, " + vocabSize + ")");
                }

                int outOffset = (b * seqLen + s) * embedDim;
                int tokenOffset = tokenId * embedDim;
                int posOffset = s * embedDim;

                for (int d = 0; d < embedDim; d++) {
                    outData[outOffset + d] = tokenEmbData[tokenOffset + d] + posEmbData[posOffset + d];
                }
            }
        }

        return new Tensor(new int[]{batch, seqLen, embedDim}, outData);
    }

    public Variable tokenEmbedding() { return tokenEmbedding; }
    public Variable positionEmbedding() { return positionEmbedding; }
    public int vocabSize() { return vocabSize; }
    public int embedDim() { return embedDim; }
    public int maxSeqLen() { return maxSeqLen; }

    public java.util.List<Variable> parameters() {
        return java.util.List.of(tokenEmbedding, positionEmbedding);
    }

    /**
     * Custom Operation for embedding lookup with autograd support.
     */
    private static final class EmbeddingOp implements com.jalllm.geforce.autograd.Operation {

        private final Variable tokenIds;
        private final Variable tokenEmb;
        private final Variable posEmb;
        private final Variable output;
        private final int vocabSize;
        private final int embedDim;
        private final int maxSeqLen;

        EmbeddingOp(Variable tokenIds, Variable tokenEmb, Variable posEmb) {
            this.tokenIds = tokenIds;
            this.tokenEmb = tokenEmb;
            this.posEmb = posEmb;
            this.vocabSize = tokenEmb.value().shape()[0];
            this.embedDim = tokenEmb.value().shape()[1];
            this.maxSeqLen = posEmb.value().shape()[0];

            Tensor result = forwardInference(tokenIds.value(), tokenEmb.value(), posEmb.value());
            this.output = Variable.fromOp(result, this, "embedding");
        }

        @Override
        public java.util.List<Variable> inputs() {
            return java.util.List.of(tokenIds, tokenEmb, posEmb);
        }

        @Override
        public Variable output() {
            return output;
        }

        @Override
        public void backward(Tensor gradOutput) {
            // Gradient w.r.t. token embeddings
            if (tokenEmb.requiresGrad()) {
                int[] tokenIdsShape = tokenIds.value().shape();
                int batch = tokenIdsShape[0];
                int seqLen = tokenIdsShape[1];

                float[] tokenIdsData = tokenIds.value().data();
                float[] gradOutData = gradOutput.data();
                float[] gradTokenEmbData = new float[vocabSize * embedDim];

                for (int b = 0; b < batch; b++) {
                    for (int s = 0; s < seqLen; s++) {
                        int tokenId = (int) tokenIdsData[b * seqLen + s];
                        if (tokenId >= 0 && tokenId < vocabSize) {
                            int gradOutOffset = (b * seqLen + s) * embedDim;
                            int tokenOffset = tokenId * embedDim;
                            for (int d = 0; d < embedDim; d++) {
                                gradTokenEmbData[tokenOffset + d] += gradOutData[gradOutOffset + d];
                            }
                        }
                    }
                }

                tokenEmb.accumulateGradient(new Tensor(new int[]{vocabSize, embedDim}, gradTokenEmbData));
            }

            // Gradient w.r.t. position embeddings
            if (posEmb.requiresGrad()) {
                int[] tokenIdsShape = tokenIds.value().shape();
                int batch = tokenIdsShape[0];
                int seqLen = tokenIdsShape[1];

                float[] gradOutData = gradOutput.data();
                float[] gradPosEmbData = new float[maxSeqLen * embedDim];

                for (int b = 0; b < batch; b++) {
                    for (int s = 0; s < seqLen; s++) {
                        int gradOutOffset = (b * seqLen + s) * embedDim;
                        int posOffset = s * embedDim;
                        for (int d = 0; d < embedDim; d++) {
                            gradPosEmbData[posOffset + d] += gradOutData[gradOutOffset + d];
                        }
                    }
                }

                posEmb.accumulateGradient(new Tensor(new int[]{maxSeqLen, embedDim}, gradPosEmbData));
            }

            // tokenIds doesn't receive gradients (discrete)
        }

        @Override
        public String name() {
            return "Embedding";
        }

        private static Tensor forwardInference(Tensor tokenIds, Tensor tokenEmb, Tensor posEmb) {
            int[] shape = tokenIds.shape();
            int batch = shape[0];
            int seqLen = shape[1];
            int embedDim = tokenEmb.shape()[1];
            int vocabSize = tokenEmb.shape()[0];
            int maxSeqLen = posEmb.shape()[0];

            if (seqLen > maxSeqLen) {
                throw new IllegalArgumentException("Sequence length " + seqLen + " exceeds maxSeqLen " + maxSeqLen);
            }

            float[] tokenIdsData = tokenIds.data();
            float[] tokenEmbData = tokenEmb.data();
            float[] posEmbData = posEmb.data();

            float[] outData = new float[batch * seqLen * embedDim];

            for (int b = 0; b < batch; b++) {
                for (int s = 0; s < seqLen; s++) {
                    int tokenId = (int) tokenIdsData[b * seqLen + s];
                    if (tokenId < 0 || tokenId >= vocabSize) {
                        throw new IllegalArgumentException("Token ID " + tokenId + " out of range [0, " + vocabSize + ")");
                    }

                    int outOffset = (b * seqLen + s) * embedDim;
                    int tokenOffset = tokenId * embedDim;
                    int posOffset = s * embedDim;

                    for (int d = 0; d < embedDim; d++) {
                        outData[outOffset + d] = tokenEmbData[tokenOffset + d] + posEmbData[posOffset + d];
                    }
                }
            }

            return new Tensor(new int[]{batch, seqLen, embedDim}, outData);
        }
    }
}