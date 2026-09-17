package com.jalllm.geforce.autograd.ops;

import com.jalllm.geforce.autograd.Operation;
import com.jalllm.geforce.autograd.Variable;
import com.jalllm.geforce.tensor.Tensor;

import java.util.List;

/** Element-wise multiplication: z = x * y */
public final class MulOp implements Operation {

    private final Variable x;
    private final Variable y;
    private final Variable z;

    public MulOp(Variable x, Variable y) {
        this.x = x;
        this.y = y;
        Tensor result = x.value().mul(y.value());
        this.z = Variable.fromOp(result, this, "mul");
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
        // dz/dx = y, dz/dy = x
        // grad_x = grad_z * y
        // grad_y = grad_z * x
        if (x.requiresGrad()) {
            x.accumulateGradient(gradOutput.mul(y.value()).sumToShape(x.value().shape()));
        }
        if (y.requiresGrad()) {
            y.accumulateGradient(gradOutput.mul(x.value()).sumToShape(y.value().shape()));
        }
    }

    @Override
    public String name() {
        return "Mul";
    }
}
