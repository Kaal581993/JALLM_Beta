package com.jalllm.geforce.tensor;

import java.util.Arrays;
import java.util.Random;

/**
 * Multi-dimensional Tensor class.
 * Backed by a contiguous float array in row-major order.
 */
public class Tensor {

    private final int[] shape;
    private final float[] data;

    public Tensor(int[] shape) {
        this.shape = shape.clone();
        int size = TensorShape.checkSizeInt(this.shape);
        this.data = new float[size];
    }

    public Tensor(int[] shape, float[] data) {
        this.shape = shape.clone();
        int size = TensorShape.checkSizeInt(this.shape);
        if (data.length != size) {
            throw new IllegalArgumentException("Data length " + data.length + " does not match shape size " + size);
        }
        // Ownership is transferred to the tensor.  This avoids an extra copy on
        // every operation; callers that need isolation should pass data.clone().
        this.data = data;
    }

    /** Convenience constructor for discrete token IDs. Tensor storage remains FP32. */
    public Tensor(int[] shape, int[] data) {
        this.shape = shape.clone();
        int size = TensorShape.checkSizeInt(this.shape);
        if (data.length != size) {
            throw new IllegalArgumentException("Data length " + data.length + " does not match shape size " + size);
        }
        this.data = new float[size];
        for (int i = 0; i < size; i++) this.data[i] = data[i];
    }

    // Static Factory Methods

    public static Tensor zeros(int... shape) {
        return new Tensor(shape);
    }

    public static Tensor ones(int... shape) {
        Tensor t = new Tensor(shape);
        t.fill(1.0f);
        return t;
    }

    public static Tensor rand(int[] shape, float min, float max) {
        Tensor t = new Tensor(shape);
        Random r = new Random();
        float[] d = t.data;
        float diff = max - min;
        for (int i = 0; i < d.length; i++) {
            d[i] = min + r.nextFloat() * diff;
        }
        return t;
    }

    public static Tensor randn(int[] shape, float mean, float std) {
        Tensor t = new Tensor(shape);
        Random r = new Random();
        float[] d = t.data;
        for (int i = 0; i < d.length; i++) {
            d[i] = mean + (float) r.nextGaussian() * std;
        }
        return t;
    }

    public static Tensor scalar(float value) {
        Tensor t = new Tensor(new int[]{1});
        t.data[0] = value;
        return t;
    }

    // Accessors and Metadata

    public int[] shape() {
        return shape.clone();
    }

    /** Returns the size of one dimension without allocating a shape copy. */
    public int dimension(int axis) {
        if (axis < 0 || axis >= shape.length) {
            throw new IndexOutOfBoundsException("Axis " + axis + " out of bounds for rank " + shape.length);
        }
        return shape[axis];
    }

    public float[] data() {
        return data;
    }

    public int rank() {
        return shape.length;
    }

    public int size() {
        return data.length;
    }

    public int elementCount() {
        return data.length;
    }

    public long numel() {
        return data.length;
    }

    public float get(int... indices) {
        int flat = TensorShape.flatIndex(shape, indices);
        return data[flat];
    }

    public void set(float value, int... indices) {
        int flat = TensorShape.flatIndex(shape, indices);
        data[flat] = value;
    }

    public void fill(float val) {
        Arrays.fill(data, val);
    }

    public Tensor copy() {
        return new Tensor(shape.clone(), data.clone());
    }

    // Reshaping and Views/Copies

    public Tensor reshape(int... newShape) {
        long currentSize = TensorShape.size(shape);
        
        // Handle single -1 dimension
        int inferIndex = -1;
        int product = 1;
        for (int i = 0; i < newShape.length; i++) {
            if (newShape[i] == -1) {
                if (inferIndex != -1) {
                    throw new IllegalArgumentException("Can only infer one dimension in reshape");
                }
                inferIndex = i;
            } else {
                product *= newShape[i];
            }
        }

        int[] resolvedShape = newShape.clone();
        if (inferIndex != -1) {
            if (currentSize % product != 0) {
                throw new IllegalArgumentException("Cannot reshape " + currentSize + " elements into " + Arrays.toString(newShape));
            }
            resolvedShape[inferIndex] = (int) (currentSize / product);
        } else {
            if (currentSize != product) {
                throw new IllegalArgumentException("Cannot reshape " + currentSize + " elements into " + Arrays.toString(newShape));
            }
        }

        return new Tensor(resolvedShape, data); // share same underlying data
    }

    // Element-wise operations

    public Tensor add(Tensor other) {
        return elementwise(other, '+');
    }

    public Tensor sub(Tensor other) {
        return elementwise(other, '-');
    }

    public Tensor mul(Tensor other) {
        return elementwise(other, '*');
    }

    public Tensor div(Tensor other) {
        return elementwise(other, '/');
    }

    public Tensor add(float val) {
        float[] res = new float[data.length];
        for (int i = 0; i < data.length; i++) {
            res[i] = this.data[i] + val;
        }
        return new Tensor(shape, res);
    }

    public Tensor sub(float val) {
        float[] res = new float[data.length];
        for (int i = 0; i < data.length; i++) {
            res[i] = this.data[i] - val;
        }
        return new Tensor(shape, res);
    }

    public Tensor mul(float val) {
        float[] res = new float[data.length];
        for (int i = 0; i < data.length; i++) {
            res[i] = this.data[i] * val;
        }
        return new Tensor(shape, res);
    }

    public Tensor div(float val) {
        float[] res = new float[data.length];
        for (int i = 0; i < data.length; i++) {
            res[i] = this.data[i] / val;
        }
        return new Tensor(shape, res);
    }

    public Tensor neg() {
        float[] res = new float[data.length];
        for (int i = 0; i < data.length; i++) {
            res[i] = -this.data[i];
        }
        return new Tensor(shape, res);
    }

    /** Sums broadcast dimensions so the result has {@code targetShape}. */
    public Tensor sumToShape(int... targetShape) {
        int[] target = targetShape.clone();
        if (target.length > shape.length) throw new IllegalArgumentException("Cannot reduce rank " + rank() + " to " + target.length);
        int offset = shape.length - target.length;
        for (int i = 0; i < shape.length; i++) {
            int targetDim = i < offset ? 1 : target[i - offset];
            if (targetDim != 1 && targetDim != shape[i]) throw new IllegalArgumentException("Cannot reduce " + TensorShape.toString(shape) + " to " + TensorShape.toString(target));
        }
        float[] out = new float[TensorShape.checkSizeInt(target)];
        for (int flat = 0; flat < data.length; flat++) {
            int remaining = flat;
            int targetFlat = 0;
            int targetStride = 1;
            for (int sourceAxis = shape.length - 1; sourceAxis >= 0; sourceAxis--) {
                int coord = remaining % shape[sourceAxis];
                remaining /= shape[sourceAxis];
                int targetAxis = sourceAxis - offset;
                if (targetAxis >= 0) {
                    int targetCoord = target[targetAxis] == 1 ? 0 : coord;
                    targetFlat += targetCoord * targetStride;
                    targetStride *= target[targetAxis];
                }
            }
            out[targetFlat] += data[flat];
        }
        return new Tensor(target, out);
    }

    private Tensor elementwise(Tensor other, char op) {
        int[] outputShape = broadcastShape(shape, other.shape);
        int outputSize = TensorShape.checkSizeInt(outputShape);
        float[] out = new float[outputSize];
        for (int flat = 0; flat < outputSize; flat++) {
            int left = broadcastFlatIndex(flat, outputShape, shape);
            int right = broadcastFlatIndex(flat, outputShape, other.shape);
            out[flat] = switch (op) {
                case '+' -> data[left] + other.data[right];
                case '-' -> data[left] - other.data[right];
                case '*' -> data[left] * other.data[right];
                case '/' -> data[left] / other.data[right];
                default -> throw new AssertionError(op);
            };
        }
        return new Tensor(outputShape, out);
    }

    private static int[] broadcastShape(int[] a, int[] b) {
        int rank = Math.max(a.length, b.length);
        int[] result = new int[rank];
        for (int i = 0; i < rank; i++) {
            int ad = i < rank - a.length ? 1 : a[i - (rank - a.length)];
            int bd = i < rank - b.length ? 1 : b[i - (rank - b.length)];
            if (ad != bd && ad != 1 && bd != 1) throw new IllegalArgumentException("Incompatible broadcast shapes: " + TensorShape.toString(a) + " and " + TensorShape.toString(b));
            result[i] = Math.max(ad, bd);
        }
        return result;
    }

    private static int broadcastFlatIndex(int outputFlat, int[] outputShape, int[] inputShape) {
        int inputAxisOffset = outputShape.length - inputShape.length;
        int inputFlat = 0;
        int inputStride = 1;
        int remaining = outputFlat;
        for (int axis = outputShape.length - 1; axis >= 0; axis--) {
            int coord = remaining % outputShape[axis];
            remaining /= outputShape[axis];
            if (axis >= inputAxisOffset) {
                int inputAxis = axis - inputAxisOffset;
                inputFlat += (inputShape[inputAxis] == 1 ? 0 : coord) * inputStride;
                inputStride *= inputShape[inputAxis];
            }
        }
        return inputFlat;
    }

    @Override
    public String toString() {
        return "Tensor(" + TensorShape.toString(shape) + ")";
    }
}
