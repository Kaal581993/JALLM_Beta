package com.jalllm.geforce.loss;

import com.jalllm.geforce.autograd.Variable;
import com.jalllm.geforce.autograd.ComputationalGraph;
import com.jalllm.geforce.tensor.Tensor;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class CrossEntropyLossTest {

    @Test
    void testForwardInference() {
        CrossEntropyLoss loss = new CrossEntropyLoss();

        // Perfect prediction: logits for class 1 are very high
        Tensor logits = new Tensor(new int[]{2, 3}, new float[]{
            100, 0, 0,   // target 0
            0, 100, 0    // target 1
        });
        Tensor targets = new Tensor(new int[]{2}, new float[]{0, 1});

        Tensor lossVal = loss.forward(logits, targets);
        assertEquals(0.0f, lossVal.data()[0], 1e-4f);
    }

    @Test
    void testForwardInferenceUniform() {
        CrossEntropyLoss loss = new CrossEntropyLoss();

        // Uniform logits -> uniform probabilities -> loss = log(vocabSize)
        Tensor logits = new Tensor(new int[]{2, 3}, new float[]{
            0, 0, 0,
            0, 0, 0
        });
        Tensor targets = new Tensor(new int[]{2}, new float[]{0, 1});

        Tensor lossVal = loss.forward(logits, targets);
        float expected = (float) Math.log(3);
        assertEquals(expected, lossVal.data()[0], 1e-4f);
    }

    @Test
    void testForwardInference3D() {
        CrossEntropyLoss loss = new CrossEntropyLoss();

        // [batch=2, seqLen=3, vocab=4]
        Tensor logits = new Tensor(new int[]{2, 3, 4});
        for (int i = 0; i < logits.data().length; i++) {
            logits.data()[i] = (float) (Math.random() * 0.1);
        }
        // Make first token in each position have high logit
        for (int b = 0; b < 2; b++) {
            for (int s = 0; s < 3; s++) {
                logits.data()[(b * 3 + s) * 4 + 0] = 10.0f;
            }
        }

        Tensor targets = new Tensor(new int[]{2, 3}, new float[]{
            0, 0, 0,
            0, 0, 0
        });

        Tensor lossVal = loss.forward(logits, targets);
        assertEquals(0.0f, lossVal.data()[0], 1e-3f);
    }

    @Test
    void testForwardWithIgnoreIndex() {
        CrossEntropyLoss loss = new CrossEntropyLoss(-1);

        Tensor logits = new Tensor(new int[]{3, 4}, new float[]{
            10, 0, 0, 0,
            0, 10, 0, 0,
            0, 0, 10, 0
        });
        Tensor targets = new Tensor(new int[]{3}, new float[]{0, 1, -1}); // Last is ignored

        Tensor lossVal = loss.forward(logits, targets);
        // Only first two contribute, each has ~0 loss
        assertEquals(0.0f, lossVal.data()[0], 1e-3f);
    }

    @Test
    void testAutograd() {
        CrossEntropyLoss loss = new CrossEntropyLoss();

        Variable logits = Variable.of(new Tensor(new int[]{2, 3}, new float[]{
            1.0f, 2.0f, 3.0f,
            4.0f, 5.0f, 6.0f
        }), "logits");

        Variable targets = Variable.of(new Tensor(new int[]{2}, new float[]{2, 1}), "targets");

        Variable lossVar = loss.forward(logits, targets);

        assertArrayEquals(new int[]{1}, lossVar.value().shape());

        // Backward
        ComputationalGraph.backward(lossVar);

        assertNotNull(logits.gradient());
        assertArrayEquals(new int[]{2, 3}, logits.gradient().shape());

        // Check gradient values
        // For first sample (target=2): softmax = [0.09, 0.24, 0.67], grad = softmax - one_hot
        // For second sample (target=1): softmax = [0.09, 0.24, 0.67], grad = softmax - one_hot
    }

    @Test
    void testAutogradIgnoreIndex() {
        CrossEntropyLoss loss = new CrossEntropyLoss(-1);

        Variable logits = Variable.of(new Tensor(new int[]{3, 4}, new float[]{
            1, 2, 3, 4,
            5, 6, 7, 8,
            9, 10, 11, 12
        }), "logits");

        Variable targets = Variable.of(new Tensor(new int[]{3}, new float[]{0, 1, -1}), "targets");

        Variable lossVar = loss.forward(logits, targets);
        ComputationalGraph.backward(lossVar);

        assertNotNull(logits.gradient());

        // Third sample should have zero gradient (ignored)
        float[] grad = logits.gradient().data();
        for (int v = 0; v < 4; v++) {
            assertEquals(0.0f, grad[2 * 4 + v], 1e-6f, "Ignored sample should have zero gradient");
        }
    }

    @Test
    void testAverageLoss() {
        CrossEntropyLoss lossAvg = new CrossEntropyLoss(-100, true);
        CrossEntropyLoss lossSum = new CrossEntropyLoss(-100, false);

        Tensor logits = new Tensor(new int[]{2, 3}, new float[]{
            10, 0, 0,
            0, 10, 0
        });
        Tensor targets = new Tensor(new int[]{2}, new float[]{0, 1});

        Tensor lossValAvg = lossAvg.forward(logits, targets);
        Tensor lossValSum = lossSum.forward(logits, targets);

        assertEquals(0.0f, lossValAvg.data()[0], 1e-3f);
        assertEquals(0.0f, lossValSum.data()[0], 1e-3f);

        // With non-zero loss
        logits = new Tensor(new int[]{2, 3}, new float[]{
            0, 0, 0,
            0, 0, 0
        });

        lossValAvg = lossAvg.forward(logits, targets);
        lossValSum = lossSum.forward(logits, targets);

        float expectedAvg = (float) Math.log(3);
        float expectedSum = 2 * (float) Math.log(3);

        assertEquals(expectedAvg, lossValAvg.data()[0], 1e-3f);
        // The sum version should return the total, not average
        assertEquals(expectedSum, lossValSum.data()[0], 1e-3f);
    }
}