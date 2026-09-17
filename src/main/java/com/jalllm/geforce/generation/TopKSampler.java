package com.jalllm.geforce.generation;

import com.jalllm.geforce.tensor.Tensor;

import java.util.Arrays;
import java.util.Random;

/**
 * Top-K sampling.
 * Samples from the top K most likely tokens.
 */
public final class TopKSampler implements Sampler {

    private final int k;
    private final float temperature;
    private final Random random;

    public TopKSampler(int k) {
        this(k, 1.0f);
    }

    public TopKSampler(int k, float temperature) {
        if (k <= 0) {
            throw new IllegalArgumentException("k must be positive");
        }
        if (temperature <= 0) {
            throw new IllegalArgumentException("Temperature must be positive");
        }
        this.k = k;
        this.temperature = temperature;
        this.random = new Random();
    }

    public TopKSampler(int k, float temperature, long seed) {
        this(k, temperature);
        this.random.setSeed(seed);
    }

    @Override
    public int sample(Tensor logits) {
        int[] shape = logits.shape();
        int vocabSize = shape[shape.length - 1];
        float[] data = logits.data();

        int offset = (logits.elementCount() / vocabSize - 1) * vocabSize;

        // Get top-k indices
        int[] indices = new int[vocabSize];
        for (int i = 0; i < vocabSize; i++) indices[i] = i;

        // Partial sort to find top-k
        int actualK = Math.min(k, vocabSize);
        for (int i = 0; i < actualK; i++) {
            int maxIdx = i;
            float maxVal = data[offset + indices[i]] / temperature;
            for (int j = i + 1; j < vocabSize; j++) {
                float val = data[offset + indices[j]] / temperature;
                if (val > maxVal) {
                    maxVal = val;
                    maxIdx = j;
                }
            }
            // Swap
            int temp = indices[i];
            indices[i] = indices[maxIdx];
            indices[maxIdx] = temp;
        }

        // Softmax over top-k
        float[] topKLogits = new float[actualK];
        for (int i = 0; i < actualK; i++) {
            topKLogits[i] = data[offset + indices[i]] / temperature;
        }

        float maxLogit = topKLogits[0];
        for (int i = 1; i < actualK; i++) {
            if (topKLogits[i] > maxLogit) maxLogit = topKLogits[i];
        }

        float sumExp = 0.0f;
        float[] probs = new float[actualK];
        for (int i = 0; i < actualK; i++) {
            probs[i] = (float) Math.exp(topKLogits[i] - maxLogit);
            sumExp += probs[i];
        }

        for (int i = 0; i < actualK; i++) {
            probs[i] /= sumExp;
        }

        // Sample
        float r = random.nextFloat();
        float cumsum = 0.0f;
        for (int i = 0; i < actualK; i++) {
            cumsum += probs[i];
            if (r <= cumsum) {
                return indices[i];
            }
        }
        return indices[actualK - 1];
    }

    @Override
    public int[] sampleBatch(Tensor logits) {
        int[] shape = logits.shape();
        int batch = shape[0];
        int vocabSize = shape[1];
        float[] data = logits.data();

        int[] results = new int[batch];

        for (int b = 0; b < batch; b++) {
            int offset = b * vocabSize;

            int[] indices = new int[vocabSize];
            for (int i = 0; i < vocabSize; i++) indices[i] = i;

            int actualK = Math.min(k, vocabSize);
            for (int i = 0; i < actualK; i++) {
                int maxIdx = i;
                float maxVal = data[offset + indices[i]] / temperature;
                for (int j = i + 1; j < vocabSize; j++) {
                    float val = data[offset + indices[j]] / temperature;
                    if (val > maxVal) {
                        maxVal = val;
                        maxIdx = j;
                    }
                }
                int temp = indices[i];
                indices[i] = indices[maxIdx];
                indices[maxIdx] = temp;
            }

            float[] topKLogits = new float[actualK];
            for (int i = 0; i < actualK; i++) {
                topKLogits[i] = data[offset + indices[i]] / temperature;
            }

            float maxLogit = topKLogits[0];
            for (int i = 1; i < actualK; i++) {
                if (topKLogits[i] > maxLogit) maxLogit = topKLogits[i];
            }

            float sumExp = 0.0f;
            float[] probs = new float[actualK];
            for (int i = 0; i < actualK; i++) {
                probs[i] = (float) Math.exp(topKLogits[i] - maxLogit);
                sumExp += probs[i];
            }

            for (int i = 0; i < actualK; i++) {
                probs[i] /= sumExp;
            }

            float r = random.nextFloat();
            float cumsum = 0.0f;
            for (int i = 0; i < actualK; i++) {
                cumsum += probs[i];
                if (r <= cumsum) {
                    results[b] = indices[i];
                    break;
                }
            }
        }
        return results;
    }

    public int k() { return k; }
    public float temperature() { return temperature; }
}