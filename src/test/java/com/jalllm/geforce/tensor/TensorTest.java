package com.jalllm.geforce.tensor;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class TensorTest {
    @Test
    void storesDataInRowMajorOrder() {
        Tensor tensor = new Tensor(new int[] {2, 3}, new float[] {1, 2, 3, 4, 5, 6});

        assertEquals(1.0f, tensor.get(0, 0));
        assertEquals(6.0f, tensor.get(1, 2));
        tensor.set(9.0f, 1, 1);
        assertEquals(9.0f, tensor.get(1, 1));
    }

    @Test
    void shapeCannotBeMutatedThroughAccessor() {
        Tensor tensor = Tensor.zeros(2, 3);
        int[] exposedShape = tensor.shape();
        exposedShape[0] = 99;

        assertArrayEquals(new int[] {2, 3}, tensor.shape());
    }

    @Test
    void reshapeSharesContiguousStorage() {
        Tensor tensor = new Tensor(new int[] {2, 3}, new float[] {1, 2, 3, 4, 5, 6});
        Tensor reshaped = tensor.reshape(3, 2);

        assertArrayEquals(new int[] {3, 2}, reshaped.shape());
        assertEquals(6.0f, reshaped.get(2, 1));
        reshaped.set(10.0f, 0, 0);
        assertEquals(10.0f, tensor.get(0, 0));
    }

    @Test
    void rejectsInvalidShapeAndIndexes() {
        assertThrows(IllegalArgumentException.class, () -> Tensor.zeros(2, -1));
        assertThrows(IndexOutOfBoundsException.class, () -> Tensor.zeros(2, 2).get(2, 0));
        assertThrows(IndexOutOfBoundsException.class, () -> TensorShape.multiIndex(new int[] {2, 2}, 4));
    }
}
