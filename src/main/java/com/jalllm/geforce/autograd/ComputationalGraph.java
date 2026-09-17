package com.jalllm.geforce.autograd;

import com.jalllm.geforce.tensor.Tensor;

import java.util.*;

/**
 * Manages the computational graph and performs reverse-mode automatic differentiation.
 * <p>
 * The graph is built dynamically during the forward pass. Each Variable created by an
 * Operation stores a reference to its creator Operation. During backward(), we traverse
 * the graph in reverse topological order, calling each Operation's backward() method.
 */
public final class ComputationalGraph {

    private ComputationalGraph() {}

    /**
     * Performs reverse-mode automatic differentiation starting from the given output variable.
     * The output variable must be a scalar (single element) or have a gradient already set.
     *
     * @param output the final output variable (typically the loss)
     */
    public static void backward(Variable output) {
        if (!output.requiresGrad()) {
            throw new IllegalArgumentException("Cannot backward through variable that doesn't require grad: " + output.name());
        }

        // If output is scalar, initialize its gradient to 1.0
        if (output.value().elementCount() == 1 && output.gradient() == null) {
            output.setGradient(Tensor.scalar(1.0f));
        } else if (output.gradient() == null) {
            throw new IllegalStateException("Output variable must have a gradient set or be a scalar. " +
                    "For non-scalar outputs, call output.setGradient(grad) before backward().");
        }

        // Topological sort: collect all operations in reverse order
        List<Operation> topoOrder = topologicalSort(output);

        // Backward pass in reverse topological order
        for (Operation op : topoOrder) {
            Variable outVar = op.output();
            Tensor gradOut = outVar.gradient();
            if (gradOut != null) {
                op.backward(gradOut);
            }
        }
    }

    /**
     * Performs topological sort of the computational graph.
     * Returns operations in order such that each operation appears before its inputs' creators.
     */
    private static List<Operation> topologicalSort(Variable output) {
        List<Operation> result = new ArrayList<>();
        Set<Operation> visited = new HashSet<>();
        Deque<Variable> stack = new ArrayDeque<>();
        stack.push(output);

        while (!stack.isEmpty()) {
            Variable v = stack.pop();
            Operation op = v.creator();
            if (op == null || visited.contains(op)) {
                continue;
            }
            visited.add(op);
            // Push inputs first (they'll be processed before this op)
            for (Variable input : op.inputs()) {
                if (input.creator() != null) {
                    stack.push(input);
                }
            }
            result.add(op);
        }

        // Operations are appended before their inputs' creators, which is already
        // reverse-topological order: each operation receives its output gradient
        // before its inputs are visited.
        return result;
    }

    /**
     * Zeroes gradients of all variables in the graph reachable from the given variable.
     */
    public static void zeroGrad(Variable root) {
        Set<Variable> visited = new HashSet<>();
        Deque<Variable> stack = new ArrayDeque<>();
        stack.push(root);

        while (!stack.isEmpty()) {
            Variable v = stack.pop();
            if (visited.contains(v)) continue;
            visited.add(v);
            v.zeroGrad();
            Operation op = v.creator();
            if (op != null) {
                for (Variable input : op.inputs()) {
                    stack.push(input);
                }
            }
        }
    }

    /**
     * Collects all leaf variables (parameters) in the graph reachable from root.
     * Useful for optimizer step.
     */
    public static List<Variable> collectParameters(Variable root) {
        List<Variable> params = new ArrayList<>();
        Set<Variable> visited = new HashSet<>();
        Deque<Variable> stack = new ArrayDeque<>();
        stack.push(root);

        while (!stack.isEmpty()) {
            Variable v = stack.pop();
            if (visited.contains(v)) continue;
            visited.add(v);
            if (v.creator() == null && v.requiresGrad()) {
                params.add(v);
            } else if (v.creator() != null) {
                for (Variable input : v.creator().inputs()) {
                    stack.push(input);
                }
            }
        }
        return params;
    }
}
