package com.jalllm.geforce.autograd;

import com.jalllm.geforce.tensor.Tensor;
import org.junit.jupiter.api.Test;

import static com.jalllm.geforce.autograd.VariableOps.*;
import static org.junit.jupiter.api.Assertions.*;

class DebugAutogradTest {

    @Test
    void testSimpleAddDebug() {
        Variable x = Variable.of(Tensor.scalar(2.0f), "x");
        Variable y = Variable.of(Tensor.scalar(3.0f), "y");
        Variable z = add(x, y);

        System.out.println("z.value: " + z.value().data()[0]);
        assertEquals(5.0f, z.value().data()[0], 1e-6f);

        System.out.println("z.requiresGrad: " + z.requiresGrad());
        System.out.println("z.creator: " + z.creator());
        System.out.println("z.gradient before backward: " + (z.gradient() == null ? "null" : z.gradient().data()[0]));

        ComputationalGraph.backward(z);

        System.out.println("z.gradient after backward: " + (z.gradient() == null ? "null" : z.gradient().data()[0]));
        System.out.println("x.gradient: " + (x.gradient() == null ? "null" : x.gradient().data()[0]));
        System.out.println("y.gradient: " + (y.gradient() == null ? "null" : y.gradient().data()[0]));

        assertEquals(1.0f, x.gradient().data()[0], 1e-6f);
        assertEquals(1.0f, y.gradient().data()[0], 1e-6f);
    }
}
