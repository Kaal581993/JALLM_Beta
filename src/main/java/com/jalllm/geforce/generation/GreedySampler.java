package com.jalllm.geforce.generation;

import com.jalllm.geforce.tensor.Tensor;

/**
 * Greedy sampling (argmax).
 * Always picks the token with highest probability.
 */
public final class GreedySampler implements Sampler {

    @Override
    public int sample(Tensor logits) {
        int[] shape = logits.shape();
        int vocabSize = shape[shape.length - 1];
        float[] data = logits.data();

        int offset = (logits.elementCount() / vocabSize - 1) * vocabSize; // Last position
        int bestIdx = 0;
        float bestVal = -Float.MAX_VALUE;

        for (int i = 0; i < vocabSize; i++) {
            if (data[offset + i] > bestVal) {
                bestVal = data[offset + i];
                bestIdx = i;
            }
        }
        return bestIdx;
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
            int bestIdx = 0;
            float bestVal = -Float.MAX_VALUE;

            for (int i = 0; i < vocabSize; i++) {
                if (data[offset + i] > bestVal) {
                    bestVal = data[offset + i];
                    bestIdx = i;
                }
            }
            results[b] = bestIdx;
        }
        return results;
    }
}