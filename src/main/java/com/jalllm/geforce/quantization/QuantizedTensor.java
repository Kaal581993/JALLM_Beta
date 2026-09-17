package com.jalllm.geforce.quantization;

import com.jalllm.geforce.tensor.Tensor;

/**
 * Container for quantized tensor data with scale and zero-point.
 */
public final class QuantizedTensor {

    private final Object data; // byte[] for INT8, byte[] for INT4 (packed)
    private final int[] shape;
    private final float scale;
    private final int zeroPoint;
    private final QuantizationType type;

    public QuantizedTensor(byte[] data, int[] shape, float scale, int zeroPoint, QuantizationType type) {
        this.data = data;
        this.shape = shape.clone();
        this.scale = scale;
        this.zeroPoint = zeroPoint;
        this.type = type;
    }

    public QuantizedTensor(byte[] data, int[] shape, float scale, int zeroPoint) {
        this(data, shape, scale, zeroPoint, QuantizationType.INT8);
    }

    public byte[] data() { return (byte[]) data; }
    public int[] shape() { return shape.clone(); }
    public float scale() { return scale; }
    public int zeroPoint() { return zeroPoint; }
    public QuantizationType type() { return type; }

    public int elementCount() {
        int count = 1;
        for (int dim : shape) count *= dim;
        return count;
    }

    @Override
    public String toString() {
        return String.format("QuantizedTensor(shape=%s, type=%s, scale=%.6f, zeroPoint=%d)",
                java.util.Arrays.toString(shape), type, scale, zeroPoint);
    }
}