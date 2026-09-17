package com.jalllm.geforce.model;

import com.jalllm.geforce.attention.CausalMask;
import com.jalllm.geforce.autograd.Variable;
import com.jalllm.geforce.autograd.ComputationalGraph;
import com.jalllm.geforce.tensor.Tensor;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class MiniGPTTest {

    @Test
    void testMiniGPTInference() {
        MiniGPT model = new MiniGPT(100, 32, 128, 2, 4); // Small model

        Tensor tokenIds = new Tensor(new int[]{2, 5}, new int[]{
            1, 2, 3, 4, 5,
            6, 7, 8, 9, 10
        });

        Tensor mask = model.createCausalMask(2, 5);

        Tensor logits = model.forward(tokenIds, mask);

        assertArrayEquals(new int[]{2, 5, 100}, logits.shape());
    }

    @Test
    void testMiniGPTAutograd() {
        MiniGPT model = new MiniGPT(50, 16, 64, 1, 2); // Very small for testing

        Variable tokenIds = Variable.of(new Tensor(new int[]{1, 3}, new int[]{1, 2, 3}), "token_ids");
        Variable mask = Variable.of(model.createCausalMask(1, 3), "mask");

        Variable logits = model.forward(tokenIds, mask);

        assertArrayEquals(new int[]{1, 3, 50}, logits.value().shape());

        // Backward
        Variable gradOut = Variable.of(Tensor.ones(1, 3, 50), "gradOut");
        logits.setGradient(gradOut.value());
        ComputationalGraph.backward(logits);

        // Check gradients
        assertNotNull(model.embedding().tokenEmbedding().gradient());
        assertNotNull(model.embedding().positionEmbedding().gradient());
        assertNotNull(model.transformer().finalLn().weight().gradient());
        assertNotNull(model.lmHeadBias().gradient());

        for (var block : model.transformer().blocks()) {
            assertNotNull(block.ln1().weight().gradient());
            assertNotNull(block.attention().wQ().gradient());
            assertNotNull(block.ln2().weight().gradient());
            assertNotNull(block.ffn().w1().gradient());
        }
    }

    @Test
    void testParameterCount() {
        MiniGPT model = new MiniGPT(1000, 128, 256, 4, 4);

        long paramCount = model.parameterCount();

        // Rough estimate:
        // Embedding: 1000*128 + 256*128 = 128000 + 32768 = 160768
        // Transformer (4 layers):
        //   Each block: 2*LayerNorm(2*128) + Attention(4*128*128 + 4*128) + FFN(2*128*512 + 2*512)
        //   = 512 + 65536 + 512 + 131072 + 1024 = ~198656 per layer
        //   4 layers = ~794624
        // Final LN: 256
        // LM Head bias: 1000
        // Total ~ 956k
        assertTrue(paramCount > 500000);
        assertTrue(paramCount < 2000000);
    }

    @Test
    void testWeightTying() {
        MiniGPT model = new MiniGPT(100, 32, 128, 2, 4);

        // LM head weight should be the same object as token embedding
        assertSame(model.embedding().tokenEmbedding(), model.lmHeadWeight());
    }

    @Test
    void testForwardShapes() {
        MiniGPT model = new MiniGPT(200, 64, 128, 2, 4);

        // Batch=1, seqLen=10
        Tensor tokenIds = new Tensor(new int[]{1, 10});
        Tensor mask = model.createCausalMask(1, 10);
        Tensor logits = model.forward(tokenIds, mask);
        assertArrayEquals(new int[]{1, 10, 200}, logits.shape());

        // Batch=4, seqLen=20
        tokenIds = new Tensor(new int[]{4, 20});
        mask = model.createCausalMask(4, 20);
        logits = model.forward(tokenIds, mask);
        assertArrayEquals(new int[]{4, 20, 200}, logits.shape());
    }
}