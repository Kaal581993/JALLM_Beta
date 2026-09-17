package com.jalllm.geforce.tensor;

import java.util.Arrays;

/**
 * Utility class for tensor shape operations.
 * A tensor shape is an array of non-negative integers describing dimensions.
 *
 * <p>For example, a tensor with shape [2, 3, 4] has 2×3×4 = 24 elements.</p>
 *
 * <p>All shapes use row-major (C-contiguous) ordering for data storage.</p>
 */
public final class TensorShape {

    private TensorShape() {
        // utility class - no instantiation
    }

    /**
     * Computes the total number of elements (product of all dimensions).
     *
     * @param shape the tensor shape
     * @return the total element count
     */
    public static long size(int[] shape) {
        long s = 1L;
        for (int d : shape) {
            if (d < 0) {
                throw new IllegalArgumentException("Negative dimension: " + Arrays.toString(shape));
            }
            s *= d;
        }
        return s;
    }

    /**
     * Returns a copy of the shape array.
     */
    public static int[] copy(int[] shape) {
        return shape.clone();
    }

    /**
     * Returns the number of dimensions (rank).
     */
    public static int rank(int[] shape) {
        return shape.length;
    }

    /**
     * Verifies that the total element count fits in an int.
     */
    public static int checkSizeInt(int[] shape) {
        long s = size(shape);
        if (s > Integer.MAX_VALUE) {
            throw new IllegalArgumentException(
                    "Tensor too large: " + s + " elements exceeds int range. Shape: " + Arrays.toString(shape));
        }
        return (int) s;
    }

    /**
     * Computes the strides for row-major (C-contiguous) layout.
     * Stride[i] = product of shape[i+1..n-1].
     */
    public static int[] strides(int[] shape) {
        int n = shape.length;
        int[] st = new int[n];
        int s = 1;
        for (int i = n - 1; i >= 0; i--) {
            st[i] = s;
            s *= shape[i];
        }
        return st;
    }

    /**
     * Computes the flat index from a multi-dimensional index using row-major strides.
     */
    public static int flatIndex(int[] shape, int[] index) {
        if (index.length != shape.length) {
            throw new IllegalArgumentException(
                    "Index rank " + index.length + " != shape rank " + shape.length);
        }
        int[] st = strides(shape);
        int flat = 0;
        for (int i = 0; i < index.length; i++) {
            if (index[i] < 0 || index[i] >= shape[i]) {
                throw new IndexOutOfBoundsException(
                        "Index " + Arrays.toString(index) + " out of bounds for shape " + Arrays.toString(shape));
            }
            flat += index[i] * st[i];
        }
        return flat;
    }

    /**
     * Converts a flat index back to a multi-dimensional index using row-major strides.
     */
    public static int[] multiIndex(int[] shape, int flat) {
        int size = checkSizeInt(shape);
        if (flat < 0 || flat >= size) {
            throw new IndexOutOfBoundsException("Flat index " + flat + " out of bounds for shape " + Arrays.toString(shape));
        }
        int[] index = new int[shape.length];
        int[] st = strides(shape);
        int remaining = flat;
        for (int i = 0; i < shape.length; i++) {
            index[i] = remaining / st[i];
            remaining %= st[i];
        }
        return index;
    }

    /**
     * Validates that two shapes are compatible for element-wise operations.
     */
    public static boolean compatible(int[] a, int[] b) {
        if (a.length != b.length) {
            return false;
        }
        for (int i = 0; i < a.length; i++) {
            if (a[i] != b[i]) {
                return false;
            }
        }
        return true;
    }

    /**
     * Returns a human-readable shape string, e.g. "[2, 3, 4]".
     */
    public static String toString(int[] shape) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < shape.length; i++) {
            if (i > 0) sb.append(", ");
            sb.append(shape[i]);
        }
        sb.append("]");
        return sb.toString();
    }
}
