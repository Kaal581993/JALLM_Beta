package com.jalllm.geforce.normalization;

import com.jalllm.geforce.autograd.Variable;
import com.jalllm.geforce.autograd.ComputationalGraph;
import com.jalllm.geforce.tensor.Tensor;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class LayerNormTest {

    @Test
    void testForwardInference() {
        LayerNorm ln = new LayerNorm(4);
        Tensor x = new Tensor(new int[]{2, 4}, new float[]{
            1, 2, 3, 4,
            10, 20, 30, 40
        });
        Tensor out = ln.forward(x);

        // Check output shape
        assertArrayEquals(new int[]{2, 4}, out.shape());

        // Check that each row has mean ~0 and std ~1 (before weight/bias)
        // Since weight=1, bias=0 initially, output should be normalized
        for (int b = 0; b < 2; b++) {
            float sum = 0;
            for (int i = 0; i < 4; i++) {
                sum += out.get(b, i);
            }
            float mean = sum / 4;
            assertEquals(0.0f, mean, 1e-5f, "Row " + b + " mean should be ~0");

            float var = 0;
            for (int i = 0; i < 4; i++) {
                float diff = out.get(b, i) - mean;
                var += diff * diff;
            }
            var /= 4;
            assertEquals(1.0f, var, 1e-5f, "Row " + b + " variance should be ~1");
        }
    }

    @Test
    void testForwardWithAutograd() {
        LayerNorm ln = new LayerNorm(3);
        Variable x = Variable.of(new Tensor(new int[]{2, 3}, new float[]{
            1, 2, 3,
            4, 5, 6
        }), "x");

        Variable out = ln.forward(x);

        assertArrayEquals(new int[]{2, 3}, out.value().shape());

        // Backward pass
        Variable gradOut = Variable.of(Tensor.ones(2, 3), "gradOut");
        out.setGradient(gradOut.value());
        ComputationalGraph.backward(out);

        assertNotNull(x.gradient());
        assertNotNull(ln.weight().gradient());
        assertNotNull(ln.bias().gradient());
        assertArrayEquals(new int[]{2, 3}, x.gradient().shape());
        assertArrayEquals(new int[]{3}, ln.weight().gradient().shape());
        assertArrayEquals(new int[]{3}, ln.bias().gradient().shape());
    }

    @Test
    void testLearnableParameters() {
        LayerNorm ln = new LayerNorm(4);

        // Weight should be initialized to 1, bias to 0
        for (int i = 0; i < 4; i++) {
            assertEquals(1.0f, ln.weight().value().get(i), 1e-6f);
            assertEquals(0.0f, ln.bias().value().get(i), 1e-6f);
        }

        // Modify weight and bias
        ln.weight().value().set(2.0f, 0);
        ln.bias().value().set(1.0f, 0);

        Tensor x = new Tensor(new int[]{1, 4}, new float[]{0, 0, 0, 0});
        Tensor out = ln.forward(x);

        // With x=0, mean=0, var=0, normalized=0
        // out = 0 * weight + bias = bias
        assertEquals(1.0f, out.get(0, 0), 1e-6f);
        assertEquals(0.0f, out.get(0, 1), 1e-6f);
        assertEquals(0.0f, out.get(0, 2), 1e-6f);
        assertEquals(0.0f, out.get(0, 3), 1e-6f);
    }

    @Test
    void test3DInput() {
        LayerNorm ln = new LayerNorm(4);
        // [batch=2, seq=3, features=4]
        Tensor x = new Tensor(new int[]{2, 3, 4}, new float[24]);
        for (int i = 0; i < 24; i++) {
            x.data()[i] = (float) (i + 1);
        }

        Tensor out = ln.forward(x);
        assertArrayEquals(new int[]{2, 3, 4}, out.shape());

        // Each [*, *, :] slice should be normalized
        for (int b = 0; b < 2; b++) {
            for (int s = 0; s < 3; s++) {
                float sum = 0;
                for (int i = 0; i < 4; i++) {
                    sum += out.get(b, s, i);
                }
                float mean = sum / 4;
                assertEquals(0.0f, mean, 1e-5f);
            }
        }
    }
}