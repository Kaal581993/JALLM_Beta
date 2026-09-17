package com.jalllm.geforce.generation;

import com.jalllm.geforce.activation.Softmax;
import com.jalllm.geforce.tensor.Tensor;

import java.util.Random;

/**
 * Temperature sampling.
 * Applies temperature to logits before softmax, then samples from the distribution.
 * Higher temperature = more random, lower temperature = more deterministic.
 */
public final class TemperatureSampler implements Sampler {

    private final float temperature;
    private final Random random;

    public TemperatureSampler(float temperature) {
        if (temperature <= 0) {
            throw new IllegalArgumentException("Temperature must be positive");
        }
        this.temperature = temperature;
        this.random = new Random();
    }

    public TemperatureSampler(float temperature, long seed) {
        this(temperature);
        this.random.setSeed(seed);
    }

    @Override
    public int sample(Tensor logits) {
        int[] shape = logits.shape();
        int vocabSize = shape[shape.length - 1];
        float[] data = logits.data();

        int offset = (logits.elementCount() / vocabSize - 1) * vocabSize;

        // Apply temperature and softmax
        float[] probs = new float[vocabSize];
        float maxLogit = -Float.MAX_VALUE;
        for (int i = 0; i < vocabSize; i++) {
            float val = data[offset + i] / temperature;
            if (val > maxLogit) maxLogit = val;
        }

        float sumExp = 0.0f;
        for (int i = 0; i < vocabSize; i++) {
            float exp = (float) Math.exp(data[offset + i] / temperature - maxLogit);
            probs[i] = exp;
            sumExp += exp;
        }

        for (int i = 0; i < vocabSize; i++) {
            probs[i] /= sumExp;
        }

        // Sample from distribution
        float r = random.nextFloat();
        float cumsum = 0.0f;
        for (int i = 0; i < vocabSize; i++) {
            cumsum += probs[i];
            if (r <= cumsum) {
                return i;
            }
        }
        return vocabSize - 1; // Fallback
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

            // Apply temperature and softmax
            float[] probs = new float[vocabSize];
            float maxLogit = -Float.MAX_VALUE;
            for (int i = 0; i < vocabSize; i++) {
                float val = data[offset + i] / temperature;
                if (val > maxLogit) maxLogit = val;
            }

            float sumExp = 0.0f;
            for (int i = 0; i < vocabSize; i++) {
                float exp = (float) Math.exp(data[offset + i] / temperature - maxLogit);
                probs[i] = exp;
                sumExp += exp;
            }

            for (int i = 0; i < vocabSize; i++) {
                probs[i] /= sumExp;
            }

            // Sample
            float r = random.nextFloat();
            float cumsum = 0.0f;
            for (int i = 0; i < vocabSize; i++) {
                cumsum += probs[i];
                if (r <= cumsum) {
                    results[b] = i;
                    break;
                }
            }
        }
        return results;
    }

    public float temperature() { return temperature; }
}