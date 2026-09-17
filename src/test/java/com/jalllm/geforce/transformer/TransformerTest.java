package com.jalllm.geforce.transformer;

import com.jalllm.geforce.attention.CausalMask;
import com.jalllm.geforce.autograd.Variable;
import com.jalllm.geforce.autograd.ComputationalGraph;
import com.jalllm.geforce.tensor.Tensor;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class TransformerTest {

    @Test
    void testFeedForwardInference() {
        FeedForward ffn = new FeedForward(16, 64);

        Tensor x = new Tensor(new int[]{2, 5, 16});
        for (int i = 0; i < x.data().length; i++) {
            x.data()[i] = (float) (Math.random() * 0.1);
        }

        Tensor out = ffn.forward(x);

        assertArrayEquals(new int[]{2, 5, 16}, out.shape());
    }

    @Test
    void testFeedForwardAutograd() {
        FeedForward ffn = new FeedForward(8, 32);

        Variable x = Variable.of(new Tensor(new int[]{1, 3, 8}), "x");
        for (int i = 0; i < x.value().data().length; i++) {
            x.value().data()[i] = (float) (Math.random() * 0.1);
        }

        Variable out = ffn.forward(x);

        assertArrayEquals(new int[]{1, 3, 8}, out.value().shape());

        // Backward
        Variable gradOut = Variable.of(Tensor.ones(1, 3, 8), "gradOut");
        out.setGradient(gradOut.value());
        ComputationalGraph.backward(out);

        assertNotNull(x.gradient());
        assertNotNull(ffn.w1().gradient());
        assertNotNull(ffn.b1().gradient());
        assertNotNull(ffn.w2().gradient());
        assertNotNull(ffn.b2().gradient());
    }

    @Test
    void testTransformerBlockInference() {
        TransformerBlock block = new TransformerBlock(16, 4);

        Tensor x = new Tensor(new int[]{2, 5, 16});
        for (int i = 0; i < x.data().length; i++) {
            x.data()[i] = (float) (Math.random() * 0.1);
        }

        Tensor mask = CausalMask.create(2, 4, 5);

        Tensor out = block.forward(x, mask);

        assertArrayEquals(new int[]{2, 5, 16}, out.shape());
    }

    @Test
    void testTransformerBlockAutograd() {
        TransformerBlock block = new TransformerBlock(8, 2);

        Variable x = Variable.of(new Tensor(new int[]{1, 3, 8}), "x");
        for (int i = 0; i < x.value().data().length; i++) {
            x.value().data()[i] = (float) (Math.random() * 0.1);
        }

        Variable mask = Variable.of(CausalMask.create(1, 2, 3), "mask");

        Variable out = block.forward(x, mask);

        assertArrayEquals(new int[]{1, 3, 8}, out.value().shape());

        // Backward
        Variable gradOut = Variable.of(Tensor.ones(1, 3, 8), "gradOut");
        out.setGradient(gradOut.value());
        ComputationalGraph.backward(out);

        assertNotNull(x.gradient());
        assertNotNull(block.ln1().weight().gradient());
        assertNotNull(block.attention().wQ().gradient());
        assertNotNull(block.ln2().weight().gradient());
        assertNotNull(block.ffn().w1().gradient());
    }

    @Test
    void testTransformerInference() {
        Transformer transformer = new Transformer(2, 16, 4); // 2 layers

        Tensor x = new Tensor(new int[]{2, 5, 16});
        for (int i = 0; i < x.data().length; i++) {
            x.data()[i] = (float) (Math.random() * 0.1);
        }

        Tensor mask = CausalMask.create(2, 4, 5);

        Tensor out = transformer.forward(x, mask);

        assertArrayEquals(new int[]{2, 5, 16}, out.shape());
    }

    @Test
    void testTransformerAutograd() {
        Transformer transformer = new Transformer(2, 8, 2); // Small for testing

        Variable x = Variable.of(new Tensor(new int[]{1, 3, 8}), "x");
        for (int i = 0; i < x.value().data().length; i++) {
            x.value().data()[i] = (float) (Math.random() * 0.1);
        }

        Variable mask = Variable.of(CausalMask.create(1, 2, 3), "mask");

        Variable out = transformer.forward(x, mask);

        assertArrayEquals(new int[]{1, 3, 8}, out.value().shape());

        // Backward
        Variable gradOut = Variable.of(Tensor.ones(1, 3, 8), "gradOut");
        out.setGradient(gradOut.value());
        ComputationalGraph.backward(out);

        assertNotNull(x.gradient());

        // Check all blocks have gradients
        for (TransformerBlock block : transformer.blocks()) {
            assertNotNull(block.ln1().weight().gradient());
            assertNotNull(block.attention().wQ().gradient());
            assertNotNull(block.ln2().weight().gradient());
            assertNotNull(block.ffn().w1().gradient());
        }
        assertNotNull(transformer.finalLn().weight().gradient());
    }

    @Test
    void testTransformerParameters() {
        Transformer transformer = new Transformer(2, 16, 4);

        var params = transformer.parameters();
        // Each block: ln1(2) + attention(8) + ln2(2) + ffn(4) = 16
        // 2 blocks = 32
        // final ln = 2
        // Total = 34
        assertEquals(34, params.size());
    }
}