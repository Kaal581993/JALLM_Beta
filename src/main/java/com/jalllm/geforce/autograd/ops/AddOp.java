package com.jalllm.geforce.autograd.ops;

import com.jalllm.geforce.autograd.Operation;
import com.jalllm.geforce.autograd.Variable;
import com.jalllm.geforce.tensor.Tensor;

import java.util.List;

/** Element-wise addition: z = x + y */
public final class AddOp implements Operation {

    private final Variable x;
    private final Variable y;
    private final Variable z;

    public AddOp(Variable x, Variable y) {
        this.x = x;
        this.y = y;
        Tensor result = x.value().add(y.value());
        this.z = Variable.fromOp(result, this, "add");
    }

    @Override
    public List<Variable> inputs() {
        return List.of(x, y);
    }

    @Override
    public Variable output() {
        return z;
    }

    @Override
    public void backward(Tensor gradOutput) {
        // dz/dx = 1, dz/dy = 1
        // grad_x = grad_z * 1 = grad_z
        // grad_y = grad_z * 1 = grad_z
        if (x.requiresGrad()) {
            x.accumulateGradient(gradOutput.sumToShape(x.value().shape()));
        }
        if (y.requiresGrad()) {
            y.accumulateGradient(gradOutput.sumToShape(y.value().shape()));
        }
    }

    @Override
    public String name() {
        return "Add";
    }
}
