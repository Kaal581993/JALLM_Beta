package com.jalllm.geforce.tensor;

/**
 * 3D specialization of Tensor.
 */
public class Tensor3D extends Tensor {

    public Tensor3D(int d1, int d2, int d3) {
        super(new int[]{d1, d2, d3});
    }

    public Tensor3D(int d1, int d2, int d3, float[] data) {
        super(new int[]{d1, d2, d3}, data);
    }

    public float get(int i, int j, int k) {
        return super.get(i, j, k);
    }

    public void set(int i, int j, int k, float val) {
        super.set(val, i, j, k);
    }

    public int d1() {
        return dimension(0);
    }

    public int d2() {
        return dimension(1);
    }

    public int d3() {
        return dimension(2);
    }
}
