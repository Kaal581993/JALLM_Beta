package com.jalllm.geforce.tensor;

/**
 * 1D specialization of Tensor.
 */
public class Tensor1D extends Tensor {

    public Tensor1D(int length) {
        super(new int[]{length});
    }

    public Tensor1D(int length, float[] data) {
        super(new int[]{length}, data);
    }

    public float get(int i) {
        return data()[i];
    }

    public void set(int i, float val) {
        data()[i] = val;
    }

    public int length() {
        return dimension(0);
    }
}
