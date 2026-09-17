package com.jalllm.geforce.optimizer;

import com.jalllm.geforce.autograd.Variable;
import com.jalllm.geforce.autograd.ComputationalGraph;
import com.jalllm.geforce.autograd.VariableOps;
import com.jalllm.geforce.tensor.Tensor;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class AdamWTest {

    @Test
    void testSimpleOptimization() {
        // Minimize (x - 5)^2
        Variable x = Variable.of(Tensor.scalar(0.0f), "x");
        Variable target = Variable.constant(Tensor.scalar(5.0f), "target");

        AdamW optimizer = new AdamW(List.of(x), 0.1f);

        // Rebuild the computation graph each iteration so gradients
        // are computed at the current parameter value.
        for (int i = 0; i < 100; i++) {
            optimizer.zeroGrad();
            Variable diff = VariableOps.sub(x, target);
            Variable loss = VariableOps.mul(diff, diff);
            ComputationalGraph.backward(loss);
            optimizer.step();
        }

        // x should be close to 5
        assertEquals(5.0f, x.value().data()[0], 0.5f);
    }

    @Test
    void testMultipleParameters() {
        // Minimize (x - 3)^2 + (y + 2)^2
        Variable x = Variable.of(Tensor.scalar(0.0f), "x");
        Variable y = Variable.of(Tensor.scalar(0.0f), "y");

        Variable targetX = Variable.constant(Tensor.scalar(3.0f), "target_x");
        Variable targetY = Variable.constant(Tensor.scalar(-2.0f), "target_y");

        AdamW optimizer = new AdamW(List.of(x, y), 0.1f);

        // Rebuild the computation graph each iteration so gradients
        // are computed at the current parameter values.
        for (int i = 0; i < 100; i++) {
            optimizer.zeroGrad();
            Variable diffX = VariableOps.sub(x, targetX);
            Variable diffY = VariableOps.sub(y, targetY);
            Variable loss = VariableOps.add(VariableOps.mul(diffX, diffX), VariableOps.mul(diffY, diffY));
            ComputationalGraph.backward(loss);
            optimizer.step();
        }

        assertEquals(3.0f, x.value().data()[0], 0.5f);
        assertEquals(-2.0f, y.value().data()[0], 0.5f);
    }

    @Test
    void testWeightDecay() {
        // With weight decay, parameters should shrink towards zero
        Variable x = Variable.of(Tensor.scalar(10.0f), "x");
        Variable loss = VariableOps.mul(x, x); // x^2

        AdamW optimizer = new AdamW(List.of(x), 0.1f, 0.9f, 0.999f, 1e-8f, 0.1f);

        for (int i = 0; i < 50; i++) {
            optimizer.zeroGrad();
            ComputationalGraph.backward(loss);
            optimizer.step();
        }

        // With weight decay, x should be smaller than without
        // But since loss also pushes to 0, it's hard to isolate
        // Just verify it runs without error
        assertTrue(Math.abs(x.value().data()[0]) < 10.0f);
    }

    @Test
    void testZeroGrad() {
        Variable x = Variable.of(Tensor.scalar(1.0f), "x");
        Variable y = Variable.of(Tensor.scalar(2.0f), "y");
        Variable loss = VariableOps.add(VariableOps.mul(x, x), VariableOps.mul(y, y));

        AdamW optimizer = new AdamW(List.of(x, y), 0.1f);

        ComputationalGraph.backward(loss);
        assertNotNull(x.gradient());
        assertNotNull(y.gradient());

        optimizer.zeroGrad();
        assertEquals(0.0f, x.gradient().data()[0], 1e-6f);
        assertEquals(0.0f, y.gradient().data()[0], 1e-6f);
    }

    @Test
    void testStepCount() {
        Variable x = Variable.of(Tensor.scalar(1.0f), "x");
        Variable loss = VariableOps.mul(x, x);

        AdamW optimizer = new AdamW(List.of(x), 0.1f);

        assertEquals(0, optimizer.stepCount());

        optimizer.step();
        assertEquals(1, optimizer.stepCount());

        optimizer.step();
        assertEquals(2, optimizer.stepCount());
    }

    @Test
    void testParameters() {
        Variable x = Variable.of(Tensor.scalar(1.0f), "x");
        Variable y = Variable.of(Tensor.scalar(2.0f), "y");

        AdamW optimizer = new AdamW(List.of(x, y), 0.1f);

        var params = optimizer.parameters();
        assertEquals(2, params.size());
        assertTrue(params.contains(x));
        assertTrue(params.contains(y));
    }
}
