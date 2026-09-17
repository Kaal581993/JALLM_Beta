package com.jalllm.geforce.tensor;

/**
 * 2D specialization of Tensor.
 */
public class Tensor2D extends Tensor {

    public Tensor2D(int rows, int cols) {
        super(new int[]{rows, cols});
    }

    public Tensor2D(int rows, int cols, float[] data) {
        super(new int[]{rows, cols}, data);
    }

    public float get(int r, int c) {
        return super.get(r, c);
    }

    public void set(int r, int c, float val) {
        super.set(val, r, c);
    }

    public int rows() {
        return dimension(0);
    }

    public int cols() {
        return dimension(1);
    }
}
