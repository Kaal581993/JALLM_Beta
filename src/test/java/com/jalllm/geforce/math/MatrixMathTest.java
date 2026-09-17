package com.jalllm.geforce.math;

import com.jalllm.geforce.tensor.Tensor;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class MatrixMathTest {
    @Test
    void multipliesMatrices() {
        Tensor result = MatrixMath.matmul(
                new Tensor(new int[] {2, 3}, new float[] {1, 2, 3, 4, 5, 6}),
                new Tensor(new int[] {3, 2}, new float[] {7, 8, 9, 10, 11, 12}));

        assertArrayEquals(new int[] {2, 2}, result.shape());
        assertEquals(58.0f, result.get(0, 0));
        assertEquals(154.0f, result.get(1, 1));
    }

    @Test
    void rejectsRankOtherThanTwoFor2DMethod() {
        assertThrows(IllegalArgumentException.class,
                () -> MatrixMath.matmul2D(Tensor.zeros(2), Tensor.zeros(2, 2)));
    }
}
