package com.jalllm.geforce.model;

import com.jalllm.geforce.attention.CausalMask;
import com.jalllm.geforce.autograd.Variable;
import com.jalllm.geforce.autograd.VariableOps;
import com.jalllm.geforce.config.ModelConfig;
import com.jalllm.geforce.embedding.TokenEmbedding;
import com.jalllm.geforce.math.MatrixMath;
import com.jalllm.geforce.transformer.Transformer;
import com.jalllm.geforce.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;

/**
 * MiniGPT - A small decoder-only Transformer language model.
 * Architecture:
 *   Token IDs -> TokenEmbedding -> Transformer -> LM Head -> Logits
 */
public final class MiniGPT implements LanguageModel {

    private final TokenEmbedding embedding;
    private final Transformer transformer;
    private final Variable lmHeadWeight;
    private final Variable lmHeadBias;
    private final int vocabSize;
    private final int embedDim;
    private final int maxSeqLen;
    private final int numLayers;
    private final int numHeads;
    private final int ffDim;
    private final float dropout;
    private final ModelConfig config;

    public MiniGPT(int vocabSize, int embedDim, int maxSeqLen, int numLayers, int numHeads, int ffDim) {
        this.vocabSize = vocabSize;
        this.embedDim = embedDim;
        this.maxSeqLen = maxSeqLen;
        this.numLayers = numLayers;
        this.numHeads = numHeads;
        this.ffDim = ffDim;
        this.dropout = 0.1f;
        this.config = new ModelConfig(embedDim, maxSeqLen, numLayers, numHeads, ffDim, vocabSize, embedDim / numHeads, 0.1f);

        // Token + Position embeddings
        this.embedding = new TokenEmbedding(vocabSize, embedDim, maxSeqLen);

        // Transformer
        this.transformer = new Transformer(numLayers, embedDim, numHeads, ffDim);

        // LM Head: projects from embedDim to vocabSize
        // Weight tying: use token embedding weights transposed
        this.lmHeadWeight = embedding.tokenEmbedding(); // Weight tying
        this.lmHeadBias = Variable.of(new Tensor(new int[]{vocabSize}), "lm_head_bias");
        com.jalllm.geforce.math.RandomInitializer.constant(lmHeadBias.value(), 0.0f);
    }

    public MiniGPT(int vocabSize, int embedDim, int maxSeqLen, int numLayers, int numHeads) {
        this(vocabSize, embedDim, maxSeqLen, numLayers, numHeads, embedDim * 4);
    }

    public MiniGPT(ModelConfig config) {
        this(config.vocabSize(), config.embedDim(), config.maxSeqLen(), config.numLayers(), config.numHeads(), config.ffDim());
    }

    @Override
    public Variable forward(Variable tokenIds, Variable mask) {
        // Embedding: [batch, seqLen] -> [batch, seqLen, embedDim]
        Variable x = embedding.forward(tokenIds);

        // Transformer: [batch, seqLen, embedDim] -> [batch, seqLen, embedDim]
        x = transformer.forward(x, mask);

        // LM Head: [batch, seqLen, embedDim] -> [batch, seqLen, vocabSize]
        // logits = x @ W^T + b (weight tying)
        Variable logits = VariableOps.add(
                VariableOps.matmul(x, VariableOps.transpose(lmHeadWeight)),
                lmHeadBias
        );

        return logits;
    }

    @Override
    public Tensor forward(Tensor tokenIds, Tensor mask) {
        // Embedding
        Tensor x = embedding.forward(tokenIds);

        // Transformer
        x = transformer.forward(x, mask);

        // LM Head
        Tensor logits = MatrixMath.matmul(x, MatrixMath.transpose(lmHeadWeight.value())).add(lmHeadBias.value());

        return logits;
    }

    @Override
    public List<Variable> parameters() {
        List<Variable> params = new ArrayList<>();
        params.addAll(embedding.parameters());
        params.addAll(transformer.parameters());
        params.add(lmHeadBias);
        // Note: lmHeadWeight is tied to embedding.tokenEmbedding(), so not added separately
        return params;
    }

    @Override
    public int vocabSize() {
        return vocabSize;
    }

    @Override
    public int embedDim() {
        return embedDim;
    }

    @Override
    public int maxSeqLen() {
        return maxSeqLen;
    }

    public int numLayers() { return numLayers; }
    public int numHeads() { return numHeads; }
    public int ffDim() { return ffDim; }
    public float dropout() { return dropout; }

    public ModelConfig config() { return config; }

    public TokenEmbedding embedding() { return embedding; }
    public Transformer transformer() { return transformer; }
    public Variable lmHeadWeight() { return lmHeadWeight; }
    public Variable lmHeadBias() { return lmHeadBias; }

    /**
     * Creates a causal mask for the given batch size and sequence length.
     */
    public Tensor createCausalMask(int batch, int seqLen) {
        return CausalMask.create(batch, numHeads, seqLen);
    }

    /**
     * Counts total parameters.
     */
    public long parameterCount() {
        long count = 0;
        for (Variable p : parameters()) {
            count += p.value().elementCount();
        }
        return count;
    }

    /**
     * Generates text autoregressively.
     */
    public int[] generate(int[] promptTokens, int maxTokens, float temperature, int topK, float topP) {
        int[] generated = new int[promptTokens.length + maxTokens];
        System.arraycopy(promptTokens, 0, generated, 0, promptTokens.length);
        int generatedLen = promptTokens.length;

        for (int step = 0; step < maxTokens; step++) {
            int currentSeqLen = Math.min(generatedLen, maxSeqLen);
            int[] inputTokens = new int[currentSeqLen];
            System.arraycopy(generated, generatedLen - currentSeqLen, inputTokens, 0, currentSeqLen);

            // Convert to float for Tensor
            float[] inputTokensFloat = new float[currentSeqLen];
            for (int i = 0; i < currentSeqLen; i++) {
                inputTokensFloat[i] = inputTokens[i];
            }

            Tensor input = new Tensor(new int[]{1, currentSeqLen}, inputTokensFloat);
            Tensor mask = createCausalMask(1, currentSeqLen);

            Tensor logits = forward(input, mask);
            // Get logits for the last position
            float[] lastLogits = new float[vocabSize];
            int lastPosOffset = (currentSeqLen - 1) * vocabSize;
            System.arraycopy(logits.data(), lastPosOffset, lastLogits, 0, vocabSize);

            // Apply temperature
            if (temperature != 1.0f) {
                for (int i = 0; i < vocabSize; i++) {
                    lastLogits[i] /= temperature;
                }
            }

            // Apply top-k
            if (topK > 0) {
                // Find top-k indices
                Integer[] indices = new Integer[vocabSize];
                for (int i = 0; i < vocabSize; i++) indices[i] = i;
                java.util.Arrays.sort(indices, (a, b) -> Float.compare(lastLogits[b], lastLogits[a]));
                float threshold = lastLogits[indices[Math.min(topK, vocabSize) - 1]];
                for (int i = 0; i < vocabSize; i++) {
                    if (lastLogits[i] < threshold) {
                        lastLogits[i] = Float.NEGATIVE_INFINITY;
                    }
                }
            }

            // Apply top-p
            if (topP < 1.0f) {
                Integer[] indices = new Integer[vocabSize];
                for (int i = 0; i < vocabSize; i++) indices[i] = i;
                java.util.Arrays.sort(indices, (a, b) -> Float.compare(lastLogits[b], lastLogits[a]));

                // Softmax
                float maxLogit = lastLogits[indices[0]];
                float sumExp = 0;
                for (int i = 0; i < vocabSize; i++) {
                    sumExp += Math.exp(lastLogits[i] - maxLogit);
                }
                float[] probs = new float[vocabSize];
                for (int i = 0; i < vocabSize; i++) {
                    probs[i] = (float) Math.exp(lastLogits[i] - maxLogit) / sumExp;
                }

                // Cumulative probability
                float cumProb = 0;
                for (int i = 0; i < vocabSize; i++) {
                    cumProb += probs[indices[i]];
                    if (cumProb > topP) {
                        for (int j = i + 1; j < vocabSize; j++) {
                            lastLogits[indices[j]] = Float.NEGATIVE_INFINITY;
                        }
                        break;
                    }
                }
            }

            // Sample
            int nextToken = sampleFromLogits(lastLogits);
            generated[generatedLen++] = nextToken;

            // Stop if we hit a special token (e.g., newline or end)
            if (nextToken == 10 || nextToken == 0) { // newline or null
                break;
            }
        }

        return java.util.Arrays.copyOf(generated, generatedLen);
    }

    private int sampleFromLogits(float[] logits) {
        // Softmax
        float maxLogit = Float.NEGATIVE_INFINITY;
        for (float l : logits) {
            if (l > maxLogit) maxLogit = l;
        }
        float sumExp = 0;
        for (float l : logits) {
            sumExp += Math.exp(l - maxLogit);
        }
        float[] probs = new float[logits.length];
        for (int i = 0; i < logits.length; i++) {
            probs[i] = (float) Math.exp(logits[i] - maxLogit) / sumExp;
        }

        // Sample
        float r = (float) Math.random();
        float cum = 0;
        for (int i = 0; i < probs.length; i++) {
            cum += probs[i];
            if (r < cum) return i;
        }
        return probs.length - 1;
    }
}