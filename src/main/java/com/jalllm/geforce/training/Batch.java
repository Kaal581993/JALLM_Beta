package com.jalllm.geforce.training;

import com.jalllm.geforce.tensor.Tensor;

/**
 * A batch of training data.
 */
public final class Batch {

    private final Tensor inputIds;
    private final Tensor targetIds;
    private final Tensor mask;

    public Batch(Tensor inputIds, Tensor targetIds, Tensor mask) {
        this.inputIds = inputIds;
        this.targetIds = targetIds;
        this.mask = mask;
    }

    public Batch(Tensor inputIds, Tensor targetIds) {
        this(inputIds, targetIds, null);
    }

    public Tensor inputIds() { return inputIds; }
    public Tensor targetIds() { return targetIds; }
    public Tensor mask() { return mask; }

    public int batchSize() { return inputIds.shape()[0]; }
    public int seqLen() { return inputIds.shape()[1]; }
}