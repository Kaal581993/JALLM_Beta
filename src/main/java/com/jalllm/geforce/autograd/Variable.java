package com.jalllm.geforce.autograd;

import com.jalllm.geforce.tensor.Tensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * A Variable wraps a Tensor and tracks its gradient for automatic differentiation.
 * Each Variable knows the Operation that created it, forming a computational graph.
 */
public final class Variable {

    private final Tensor value;
    private Tensor gradient;
    private final Operation creator;
    private final boolean requiresGrad;
    private final String name;

    private Variable(Tensor value, Operation creator, boolean requiresGrad, String name) {
        this.value = value;
        this.creator = creator;
        this.requiresGrad = requiresGrad;
        this.name = name;
        // Initialize gradient to null - will be created on first backward pass
        this.gradient = null;
    }

    /**
     * Creates a leaf variable (input to the graph) that requires gradients.
     */
    public static Variable of(Tensor value, String name) {
        return new Variable(value, null, true, name);
    }

    /**
     * Creates a leaf variable that does not require gradients (e.g., targets).
     */
    public static Variable constant(Tensor value, String name) {
        return new Variable(value, null, false, name);
    }

    /**
     * Creates a variable from an operation result.
     */
    public static Variable fromOp(Tensor value, Operation creator, String name) {
        return new Variable(value, creator, true, name);
    }

    public Tensor value() {
        return value;
    }

    public Tensor gradient() {
        return gradient;
    }

    public void setGradient(Tensor gradient) {
        if (!requiresGrad) {
            throw new IllegalStateException("Cannot set gradient on variable that doesn't require grad: " + name);
        }
        if (!com.jalllm.geforce.tensor.TensorShape.compatible(this.value.shape(), gradient.shape())) {
            throw new IllegalArgumentException("Gradient shape mismatch: " +
                    com.jalllm.geforce.tensor.TensorShape.toString(this.value.shape()) + " vs " +
                    com.jalllm.geforce.tensor.TensorShape.toString(gradient.shape()));
        }
        this.gradient = gradient;
    }

    public void accumulateGradient(Tensor grad) {
        if (!requiresGrad) return;
        if (this.gradient == null) {
            this.gradient = grad.copy();
        } else {
            this.gradient = this.gradient.add(grad);
        }
    }

    public boolean requiresGrad() {
        return requiresGrad;
    }

    public Operation creator() {
        return creator;
    }

    public String name() {
        return name;
    }

    public void zeroGrad() {
        if (requiresGrad) {
            if (this.gradient != null) {
                Arrays.fill(this.gradient.data(), 0.0f);
            }
        }
    }

    @Override
    public String toString() {
        return "Variable(" + name + ", shape=" + com.jalllm.geforce.tensor.TensorShape.toString(value.shape()) +
                ", requiresGrad=" + requiresGrad + ")";
    }
}