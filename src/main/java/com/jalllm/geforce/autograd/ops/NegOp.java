package com.jalllm.geforce.autograd.ops;

import com.jalllm.geforce.autograd.Operation;
import com.jalllm.geforce.autograd.Variable;
import com.jalllm.geforce.tensor.Tensor;

import java.util.List;

/** Element-wise negation: z = -x */
public final class NegOp implements Operation {

    private final Variable x;
    private final Variable z;

    public NegOp(Variable x) {
        this.x = x;
        Tensor result = x.value().neg();
        this.z = Variable.fromOp(result, this, "neg");
    }

    @Override
    public List<Variable> inputs() {
        return List.of(x);
    }

    @Override
    public Variable output() {
        return z;
    }

    @Override
    public void backward(Tensor gradOutput) {
        // dz/dx = -1
        if (x.requiresGrad()) {
            x.accumulateGradient(gradOutput.neg());
        }
    }

    @Override
    public String name() {
        return "Neg";
    }
}