package com.jalllm.geforce.optimizer;

import com.jalllm.geforce.autograd.Variable;

import java.util.List;

/**
 * Base interface for optimizers.
 */
public interface Optimizer {

    /**
     * Performs a single optimization step.
     * Updates parameters based on their gradients.
     */
    void step();

    /**
     * Zeroes the gradients of all parameters.
     */
    void zeroGrad();

    /**
     * Returns the learning rate.
     */
    float learningRate();

    /**
     * Sets the learning rate.
     */
    void setLearningRate(float lr);

    /**
     * Returns the parameters being optimized.
     */
    List<Variable> parameters();
}