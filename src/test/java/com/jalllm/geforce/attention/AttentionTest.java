package com.jalllm.geforce.attention;

import com.jalllm.geforce.autograd.Variable;
import com.jalllm.geforce.autograd.ComputationalGraph;
import com.jalllm.geforce.tensor.Tensor;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class AttentionTest {

    @Test
    void testCausalMask() {
        Tensor mask = CausalMask.create(4);

        assertArrayEquals(new int[]{4, 4}, mask.shape());

        // Lower triangular (including diagonal) should be 0
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j <= i; j++) {
                assertEquals(0.0f, mask.get(i, j), 1e-6f, "Position (" + i + "," + j + ") should be 0");
            }
        }

        // Upper triangular should be -inf
        for (int i = 0; i < 4; i++) {
            for (int j = i + 1; j < 4; j++) {
                assertEquals(Float.NEGATIVE_INFINITY, mask.get(i, j), "Position (" + i + "," + j + ") should be -inf");
            }
        }
    }

    @Test
    void testCausalMaskWithBatchAndHeads() {
        Tensor mask = CausalMask.create(2, 3, 4);

        assertArrayEquals(new int[]{2, 3, 4, 4}, mask.shape());

        for (int b = 0; b < 2; b++) {
            for (int h = 0; h < 3; h++) {
                for (int i = 0; i < 4; i++) {
                    for (int j = 0; j <= i; j++) {
                        assertEquals(0.0f, mask.get(b, h, i, j), 1e-6f);
                    }
                    for (int j = i + 1; j < 4; j++) {
                        assertEquals(Float.NEGATIVE_INFINITY, mask.get(b, h, i, j));
                    }
                }
            }
        }
    }

    @Test
    void testCausalMaskApply() {
        Tensor scores = Tensor.ones(2, 3, 4, 4);
        CausalMask.apply(scores);

        for (int b = 0; b < 2; b++) {
            for (int h = 0; h < 3; h++) {
                for (int i = 0; i < 4; i++) {
                    for (int j = 0; j <= i; j++) {
                        assertEquals(1.0f, scores.get(b, h, i, j), 1e-6f);
                    }
                    for (int j = i + 1; j < 4; j++) {
                        assertEquals(Float.NEGATIVE_INFINITY, scores.get(b, h, i, j));
                    }
                }
            }
        }
    }

    @Test
    void testScaledDotProductAttentionInference() {
        // Simple test: Q=K=V=identity-like
        Tensor q = new Tensor(new int[]{1, 2, 3}, new float[]{
            1, 0, 0,
            0, 1, 0
        });
        Tensor k = q.copy();
        Tensor v = q.copy();

        Tensor out = ScaledDotProductAttention.forward(q, k, v, null);

        assertArrayEquals(new int[]{1, 2, 3}, out.shape());
    }

    @Test
    void testScaledDotProductAttentionWithMask() {
        Tensor q = Tensor.ones(1, 2, 3);
        Tensor k = Tensor.ones(1, 2, 3);
        Tensor v = Tensor.ones(1, 2, 3);

        // Causal mask
        Tensor mask = CausalMask.create(2);

        Tensor out = ScaledDotProductAttention.forward(q, k, v, mask);

        assertArrayEquals(new int[]{1, 2, 3}, out.shape());
    }

    @Test
    void testMultiHeadAttentionInference() {
        MultiHeadAttention mha = new MultiHeadAttention(16, 4); // embedDim=16, numHeads=4, headDim=4

        Tensor x = new Tensor(new int[]{2, 5, 16}); // batch=2, seqLen=5, embedDim=16
        for (int i = 0; i < x.data().length; i++) {
            x.data()[i] = (float) (Math.random() * 0.1);
        }

        Tensor mask = CausalMask.create(2, 4, 5); // batch=2, heads=4, seqLen=5

        Tensor out = mha.forward(x, mask);

        assertArrayEquals(new int[]{2, 5, 16}, out.shape());
    }

    @Test
    void testMultiHeadAttentionAutograd() {
        MultiHeadAttention mha = new MultiHeadAttention(8, 2); // Small for testing

        Variable x = Variable.of(new Tensor(new int[]{1, 3, 8}), "x");
        for (int i = 0; i < x.value().data().length; i++) {
            x.value().data()[i] = (float) (Math.random() * 0.1);
        }

        Variable mask = Variable.of(CausalMask.create(1, 2, 3), "mask");

        Variable out = mha.forward(x, mask);

        assertArrayEquals(new int[]{1, 3, 8}, out.value().shape());

        // Backward pass
        Variable gradOut = Variable.of(Tensor.ones(1, 3, 8), "gradOut");
        out.setGradient(gradOut.value());
        ComputationalGraph.backward(out);

        // Check gradients exist
        assertNotNull(x.gradient());
        assertNotNull(mha.wQ().gradient());
        assertNotNull(mha.wK().gradient());
        assertNotNull(mha.wV().gradient());
        assertNotNull(mha.wO().gradient());
        assertNotNull(mha.bQ().gradient());
        assertNotNull(mha.bK().gradient());
        assertNotNull(mha.bV().gradient());
        assertNotNull(mha.bO().gradient());
    }

    @Test
    void testMultiHeadAttentionParameters() {
        MultiHeadAttention mha = new MultiHeadAttention(16, 4);

        var params = mha.parameters();
        assertEquals(8, params.size()); // wQ, wK, wV, wO, bQ, bK, bV, bO

        // Check shapes
        assertArrayEquals(new int[]{16, 16}, mha.wQ().value().shape());
        assertArrayEquals(new int[]{16, 16}, mha.wK().value().shape());
        assertArrayEquals(new int[]{16, 16}, mha.wV().value().shape());
        assertArrayEquals(new int[]{16, 16}, mha.wO().value().shape());
        assertArrayEquals(new int[]{16}, mha.bQ().value().shape());
        assertArrayEquals(new int[]{16}, mha.bK().value().shape());
        assertArrayEquals(new int[]{16}, mha.bV().value().shape());
        assertArrayEquals(new int[]{16}, mha.bO().value().shape());
    }
}