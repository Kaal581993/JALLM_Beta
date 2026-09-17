package com.jalllm.geforce.generation;

import com.jalllm.geforce.tensor.Tensor;

/**
 * Base interface for sampling strategies.
 */
public interface Sampler {

    /**
     * Samples a token from the probability distribution.
     *
     * @param logits Logits for the next token [vocabSize] or [batch, vocabSize]
     * @return Sampled token ID
     */
    int sample(Tensor logits);

    /**
     * Samples a token from the probability distribution (batched).
     *
     * @param logits Logits for the next token [batch, vocabSize]
     * @return Array of sampled token IDs [batch]
     */
    int[] sampleBatch(Tensor logits);
}