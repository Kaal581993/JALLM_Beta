package com.jalllm.geforce.autograd;

import com.jalllm.geforce.tensor.Tensor;
import org.junit.jupiter.api.Test;

import static com.jalllm.geforce.autograd.VariableOps.*;
import static org.junit.jupiter.api.Assertions.*;

class AutogradTest {

    @Test
    void testSimpleAdd() {
        Variable x = Variable.of(Tensor.scalar(2.0f), "x");
        Variable y = Variable.of(Tensor.scalar(3.0f), "y");
        Variable z = add(x, y);

        assertEquals(5.0f, z.value().data()[0], 1e-6f);

        ComputationalGraph.backward(z);

        assertEquals(1.0f, x.gradient().data()[0], 1e-6f);
        assertEquals(1.0f, y.gradient().data()[0], 1e-6f);
    }

    @Test
    void testSimpleMul() {
        Variable x = Variable.of(Tensor.scalar(2.0f), "x");
        Variable y = Variable.of(Tensor.scalar(3.0f), "y");
        Variable z = mul(x, y);

        assertEquals(6.0f, z.value().data()[0], 1e-6f);

        ComputationalGraph.backward(z);

        assertEquals(3.0f, x.gradient().data()[0], 1e-6f);
        assertEquals(2.0f, y.gradient().data()[0], 1e-6f);
    }

    @Test
    void testChainRule() {
        // z = (x + y) * w
        Variable x = Variable.of(Tensor.scalar(2.0f), "x");
        Variable y = Variable.of(Tensor.scalar(3.0f), "y");
        Variable w = Variable.of(Tensor.scalar(4.0f), "w");

        Variable sum = add(x, y);      // 5
        Variable z = mul(sum, w);      // 20

        assertEquals(20.0f, z.value().data()[0], 1e-6f);

        ComputationalGraph.backward(z);

        // dz/dx = w * 1 = 4
        // dz/dy = w * 1 = 4
        // dz/dw = (x + y) = 5
        assertEquals(4.0f, x.gradient().data()[0], 1e-6f);
        assertEquals(4.0f, y.gradient().data()[0], 1e-6f);
        assertEquals(5.0f, w.gradient().data()[0], 1e-6f);
    }

    @Test
    void testMatMul() {
        Variable x = Variable.of(new Tensor(new int[]{2, 2}, new float[]{1, 2, 3, 4}), "x");
        Variable y = Variable.of(new Tensor(new int[]{2, 2}, new float[]{5, 6, 7, 8}), "y");
        Variable z = matmul(x, y);

        // [1 2] @ [5 6] = [19 22]
        // [3 4]   [7 8]   [43 50]
        assertEquals(19.0f, z.value().get(0, 0), 1e-6f);
        assertEquals(22.0f, z.value().get(0, 1), 1e-6f);
        assertEquals(43.0f, z.value().get(1, 0), 1e-6f);
        assertEquals(50.0f, z.value().get(1, 1), 1e-6f);

        // Gradient: all ones
        Variable gradOut = Variable.of(Tensor.ones(2, 2), "gradOut");
        z.setGradient(gradOut.value());
        ComputationalGraph.backward(z);

        // grad_x = grad_z @ y^T
        // grad_y = x^T @ grad_z
        Tensor expectedGradX = new Tensor(new int[]{2, 2}, new float[]{13, 13, 13, 13}); // [1 1] @ [5 7] = [12 14] -> wait
        // Actually: grad_z = [[1,1],[1,1]], y^T = [[5,7],[6,8]]
        // grad_x = [[1,1],[1,1]] @ [[5,7],[6,8]] = [[11,15],[11,15]]
        assertEquals(11.0f, x.gradient().get(0, 0), 1e-6f);
        assertEquals(15.0f, x.gradient().get(0, 1), 1e-6f);
        assertEquals(11.0f, x.gradient().get(1, 0), 1e-6f);
        assertEquals(15.0f, x.gradient().get(1, 1), 1e-6f);

        // grad_y = x^T @ grad_z = [[1,3],[2,4]] @ [[1,1],[1,1]] = [[4,4],[6,6]]
        assertEquals(4.0f, y.gradient().get(0, 0), 1e-6f);
        assertEquals(4.0f, y.gradient().get(0, 1), 1e-6f);
        assertEquals(6.0f, y.gradient().get(1, 0), 1e-6f);
        assertEquals(6.0f, y.gradient().get(1, 1), 1e-6f);
    }

    @Test
    void testGelu() {
        Variable x = Variable.of(Tensor.scalar(1.0f), "x");
        Variable z = gelu(x);

        // GELU(1) ≈ 0.841
        assertEquals(0.841f, z.value().data()[0], 1e-3f);

        ComputationalGraph.backward(z);

        // Gradient should be non-zero
        assertNotEquals(0.0f, x.gradient().data()[0]);
    }

    @Test
    void testSilu() {
        Variable x = Variable.of(Tensor.scalar(1.0f), "x");
        Variable z = silu(x);

        // SiLU(1) = 1 * sigmoid(1) ≈ 0.731
        assertEquals(0.731f, z.value().data()[0], 1e-3f);

        ComputationalGraph.backward(z);

        assertNotEquals(0.0f, x.gradient().data()[0]);
    }

    @Test
    void testSoftmax() {
        Variable x = Variable.of(new Tensor(new int[]{3}, new float[]{1.0f, 2.0f, 3.0f}), "x");
        Variable z = softmax(x);

        float sum = z.value().get(0) + z.value().get(1) + z.value().get(2);
        assertEquals(1.0f, sum, 1e-6f);

        // Gradient test
        Variable gradOut = Variable.of(Tensor.ones(3), "gradOut");
        z.setGradient(gradOut.value());
        ComputationalGraph.backward(z);

        // Gradient should be computed
        assertNotNull(x.gradient());
    }

    @Test
    void testReshape() {
        Variable x = Variable.of(new Tensor(new int[]{2, 3}, new float[]{1, 2, 3, 4, 5, 6}), "x");
        Variable z = reshape(x, 3, 2);

        assertArrayEquals(new int[]{3, 2}, z.value().shape());
        assertEquals(6.0f, z.value().get(2, 1), 1e-6f);

        Variable gradOut = Variable.of(Tensor.ones(3, 2), "gradOut");
        z.setGradient(gradOut.value());
        ComputationalGraph.backward(z);

        assertNotNull(x.gradient());
        assertArrayEquals(new int[]{2, 3}, x.gradient().shape());
    }

    @Test
    void testSum() {
        Variable x = Variable.of(new Tensor(new int[]{3}, new float[]{1.0f, 2.0f, 3.0f}), "x");
        Variable z = sum(x);

        assertEquals(6.0f, z.value().data()[0], 1e-6f);

        ComputationalGraph.backward(z);

        // Gradient of sum is 1 for each element
        assertEquals(1.0f, x.gradient().get(0), 1e-6f);
        assertEquals(1.0f, x.gradient().get(1), 1e-6f);
        assertEquals(1.0f, x.gradient().get(2), 1e-6f);
    }

    @Test
    void testMean() {
        Variable x = Variable.of(new Tensor(new int[]{3}, new float[]{1.0f, 2.0f, 3.0f}), "x");
        Variable z = mean(x);

        assertEquals(2.0f, z.value().data()[0], 1e-6f);

        ComputationalGraph.backward(z);

        // Gradient of mean is 1/n for each element
        assertEquals(1.0f/3.0f, x.gradient().get(0), 1e-6f);
        assertEquals(1.0f/3.0f, x.gradient().get(1), 1e-6f);
        assertEquals(1.0f/3.0f, x.gradient().get(2), 1e-6f);
    }

    @Test
    void testZeroGrad() {
        Variable x = Variable.of(Tensor.scalar(2.0f), "x");
        Variable y = Variable.of(Tensor.scalar(3.0f), "y");
        Variable z = mul(x, y);

        ComputationalGraph.backward(z);
        assertEquals(3.0f, x.gradient().data()[0], 1e-6f);

        ComputationalGraph.zeroGrad(z);
        assertEquals(0.0f, x.gradient().data()[0], 1e-6f);
        assertEquals(0.0f, y.gradient().data()[0], 1e-6f);
    }

    @Test
    void testCollectParameters() {
        Variable x = Variable.of(Tensor.scalar(2.0f), "x");
        Variable y = Variable.of(Tensor.scalar(3.0f), "y");
        Variable z = mul(x, y);

        var params = ComputationalGraph.collectParameters(z);
        assertEquals(2, params.size());
        assertTrue(params.contains(x));
        assertTrue(params.contains(y));
    }
}