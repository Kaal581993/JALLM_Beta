package com.jalllm.geforce.autograd.ops;

import com.jalllm.geforce.autograd.Operation;
import com.jalllm.geforce.autograd.Variable;
import com.jalllm.geforce.tensor.Tensor;

import java.util.List;

/** Element-wise division: z = x / y */
public final class DivOp implements Operation {

    private final Variable x;
    private final Variable y;
    private final Variable z;

    public DivOp(Variable x, Variable y) {
        this.x = x;
        this.y = y;
        Tensor result = x.value().div(y.value());
        this.z = Variable.fromOp(result, this, "div");
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
        // z = x / y
        // dz/dx = 1/y
        // dz/dy = -x/y^2
        if (x.requiresGrad()) {
            x.accumulateGradient(gradOutput.div(y.value()).sumToShape(x.value().shape()));
        }
        if (y.requiresGrad()) {
            Tensor ySq = y.value().mul(y.value());
            Tensor negXDivYSq = x.value().div(ySq).neg();
            y.accumulateGradient(gradOutput.mul(negXDivYSq).sumToShape(y.value().shape()));
        }
    }

    @Override
    public String name() {
        return "Div";
    }
}
