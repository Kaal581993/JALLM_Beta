package com.jalllm.geforce.attention;

import com.jalllm.geforce.tensor.Tensor;

/**
 * Causal (autoregressive) attention mask.
 * Prevents positions from attending to future positions.
 * <p>
 * For a sequence of length L, the mask is an L×L matrix where:
 * mask[i][j] = 0 if i >= j (can attend)
 * mask[i][j] = -inf if i < j (cannot attend)
 */
public final class CausalMask {

    private CausalMask() {}

    /**
     * Creates a causal mask for the given sequence length.
     * Returns a 2D tensor of shape [seqLen, seqLen] with 0 for allowed positions
     * and Float.NEGATIVE_INFINITY for masked positions.
     */
    public static Tensor create(int seqLen) {
        float[] data = new float[seqLen * seqLen];
        for (int i = 0; i < seqLen; i++) {
            for (int j = 0; j < seqLen; j++) {
                if (i < j) {
                    data[i * seqLen + j] = Float.NEGATIVE_INFINITY;
                } else {
                    data[i * seqLen + j] = 0.0f;
                }
            }
        }
        return new Tensor(new int[]{seqLen, seqLen}, data);
    }

    /**
     * Creates a causal mask with batch and head dimensions.
     * Returns a 4D tensor of shape [batch, heads, seqLen, seqLen].
     */
    public static Tensor create(int batch, int heads, int seqLen) {
        float[] data = new float[batch * heads * seqLen * seqLen];
        for (int b = 0; b < batch; b++) {
            for (int h = 0; h < heads; h++) {
                int base = (b * heads + h) * seqLen * seqLen;
                for (int i = 0; i < seqLen; i++) {
                    for (int j = 0; j < seqLen; j++) {
                        if (i < j) {
                            data[base + i * seqLen + j] = Float.NEGATIVE_INFINITY;
                        } else {
                            data[base + i * seqLen + j] = 0.0f;
                        }
                    }
                }
            }
        }
        return new Tensor(new int[]{batch, heads, seqLen, seqLen}, data);
    }

    /**
     * Applies causal mask to attention scores in-place.
     * scores shape: [..., seqLen, seqLen]
     */
    public static void apply(Tensor scores) {
        int[] shape = scores.shape();
        int rank = shape.length;
        if (rank < 2) {
            throw new IllegalArgumentException("Scores must have at least 2 dimensions");
        }
        int seqLen = shape[rank - 1];
        if (shape[rank - 2] != seqLen) {
            throw new IllegalArgumentException("Last two dimensions must be equal for causal mask");
        }

        int outerSize = 1;
        for (int i = 0; i < rank - 2; i++) {
            outerSize *= shape[i];
        }

        float[] data = scores.data();
        for (int b = 0; b < outerSize; b++) {
            int offset = b * seqLen * seqLen;
            for (int i = 0; i < seqLen; i++) {
                for (int j = 0; j < seqLen; j++) {
                    if (i < j) {
                        data[offset + i * seqLen + j] = Float.NEGATIVE_INFINITY;
                    }
                }
            }
        }
    }

    /**
     * Creates a causal mask for variable sequence lengths (for packed sequences).
     * mask[i][j] = 0 if i >= j, else -inf
     */
    public static Tensor createVariableLength(int maxSeqLen, int[] actualLengths) {
        int batch = actualLengths.length;
        float[] data = new float[batch * maxSeqLen * maxSeqLen];

        for (int b = 0; b < batch; b++) {
            int len = actualLengths[b];
            int base = b * maxSeqLen * maxSeqLen;
            for (int i = 0; i < maxSeqLen; i++) {
                for (int j = 0; j < maxSeqLen; j++) {
                    if (i >= len || j >= len) {
                        // Padding positions: mask everything
                        data[base + i * maxSeqLen + j] = Float.NEGATIVE_INFINITY;
                    } else if (i < j) {
                        // Future positions: mask
                        data[base + i * maxSeqLen + j] = Float.NEGATIVE_INFINITY;
                    } else {
                        // Valid past/present positions: allow
                        data[base + i * maxSeqLen + j] = 0.0f;
                    }
                }
            }
        }

        return new Tensor(new int[]{batch, maxSeqLen, maxSeqLen}, data);
    }
}