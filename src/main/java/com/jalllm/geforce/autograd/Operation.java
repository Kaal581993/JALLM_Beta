package com.jalllm.geforce.autograd;

import com.jalllm.geforce.tensor.Tensor;

import java.util.List;

/**
 * Represents a differentiable operation in the computational graph.
 * Each operation knows how to compute its forward pass and backward pass.
 */
public interface Operation {

    /**
     * Returns the input variables to this operation.
     */
    List<Variable> inputs();

    /**
     * Returns the output variable of this operation.
     */
    Variable output();

    /**
     * Performs the backward pass.
     * Given the gradient of the loss with respect to the output,
     * computes and accumulates gradients for all inputs.
     *
     * @param gradOutput gradient of loss w.r.t. this operation's output
     */
    void backward(Tensor gradOutput);

    /**
     * Human-readable name for debugging.
     */
    String name();
}