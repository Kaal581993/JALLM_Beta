package com.jalllm.geforce.generation;

import com.jalllm.geforce.tensor.Tensor;

import java.util.Arrays;
import java.util.Random;

/**
 * Top-P (Nucleus) sampling.
 * Samples from the smallest set of tokens whose cumulative probability exceeds p.
 */
public final class TopPSampler implements Sampler {

    private final float p;
    private final float temperature;
    private final Random random;

    public TopPSampler(float p) {
        this(p, 1.0f);
    }

    public TopPSampler(float p, float temperature) {
        if (p <= 0 || p > 1) {
            throw new IllegalArgumentException("p must be in (0, 1]");
        }
        if (temperature <= 0) {
            throw new IllegalArgumentException("Temperature must be positive");
        }
        this.p = p;
        this.temperature = temperature;
        this.random = new Random();
    }

    public TopPSampler(float p, float temperature, long seed) {
        this(p, temperature);
        this.random.setSeed(seed);
    }

    @Override
    public int sample(Tensor logits) {
        int[] shape = logits.shape();
        int vocabSize = shape[shape.length - 1];
        float[] data = logits.data();

        int offset = (logits.elementCount() / vocabSize - 1) * vocabSize;

        // Create index array and sort by logit value (descending)
        Integer[] indices = new Integer[vocabSize];
        for (int i = 0; i < vocabSize; i++) indices[i] = i;

        // Sort indices by logit value (descending)
        Arrays.sort(indices, (a, b) -> Float.compare(
                data[offset + b] / temperature,
                data[offset + a] / temperature
        ));

        // Compute softmax probabilities in sorted order
        float[] sortedLogits = new float[vocabSize];
        for (int i = 0; i < vocabSize; i++) {
            sortedLogits[i] = data[offset + indices[i]] / temperature;
        }

        float maxLogit = sortedLogits[0];
        float[] probs = new float[vocabSize];
        float sumExp = 0.0f;
        for (int i = 0; i < vocabSize; i++) {
            probs[i] = (float) Math.exp(sortedLogits[i] - maxLogit);
            sumExp += probs[i];
        }
        for (int i = 0; i < vocabSize; i++) {
            probs[i] /= sumExp;
        }

        // Find nucleus (top-p)
        float cumsum = 0.0f;
        int nucleusSize = vocabSize;
        for (int i = 0; i < vocabSize; i++) {
            cumsum += probs[i];
            if (cumsum >= p) {
                nucleusSize = i + 1;
                break;
            }
        }

        // Renormalize probabilities within nucleus
        float nucleusSum = 0.0f;
        for (int i = 0; i < nucleusSize; i++) {
            nucleusSum += probs[i];
        }
        for (int i = 0; i < nucleusSize; i++) {
            probs[i] /= nucleusSum;
        }

        // Sample from nucleus
        float r = random.nextFloat();
        cumsum = 0.0f;
        for (int i = 0; i < nucleusSize; i++) {
            cumsum += probs[i];
            if (r <= cumsum) {
                return indices[i];
            }
        }
        return indices[nucleusSize - 1];
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

            Integer[] indices = new Integer[vocabSize];
            for (int i = 0; i < vocabSize; i++) indices[i] = i;

            Arrays.sort(indices, (a, bIdx) -> Float.compare(
                    data[offset + bIdx] / temperature,
                    data[offset + a] / temperature
            ));

            float[] sortedLogits = new float[vocabSize];
            for (int i = 0; i < vocabSize; i++) {
                sortedLogits[i] = data[offset + indices[i]] / temperature;
            }

            float maxLogit = sortedLogits[0];
            float[] probs = new float[vocabSize];
            float sumExp = 0.0f;
            for (int i = 0; i < vocabSize; i++) {
                probs[i] = (float) Math.exp(sortedLogits[i] - maxLogit);
                sumExp += probs[i];
            }
            for (int i = 0; i < vocabSize; i++) {
                probs[i] /= sumExp;
            }

            float cumsum = 0.0f;
            int nucleusSize = vocabSize;
            for (int i = 0; i < vocabSize; i++) {
                cumsum += probs[i];
                if (cumsum >= p) {
                    nucleusSize = i + 1;
                    break;
                }
            }

            float nucleusSum = 0.0f;
            for (int i = 0; i < nucleusSize; i++) {
                nucleusSum += probs[i];
            }
            for (int i = 0; i < nucleusSize; i++) {
                probs[i] /= nucleusSum;
            }

            float r = random.nextFloat();
            cumsum = 0.0f;
            for (int i = 0; i < nucleusSize; i++) {
                cumsum += probs[i];
                if (r <= cumsum) {
                    results[b] = indices[i];
                    break;
                }
            }
        }
        return results;
    }

    public float p() { return p; }
    public float temperature() { return temperature; }
}