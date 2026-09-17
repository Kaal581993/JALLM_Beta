package com.jalllm.geforce.activation;

import com.jalllm.geforce.tensor.Tensor;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class SoftmaxTest {
    @Test
    void isStableForLargeLogitsAndNormalizesEachRow() {
        Tensor output = Softmax.forward(new Tensor(new int[] {2, 3},
                new float[] {1000, 1001, 1002, -1000, -1001, -1002}));

        assertEquals(1.0f, output.get(0, 0) + output.get(0, 1) + output.get(0, 2), 1e-6f);
        assertEquals(1.0f, output.get(1, 0) + output.get(1, 1) + output.get(1, 2), 1e-6f);
    }

    @Test
    void backwardRejectsMismatchedGradientShape() {
        assertThrows(IllegalArgumentException.class,
                () -> Softmax.backward(Tensor.zeros(2, 3), Tensor.zeros(3, 2)));
    }
}
