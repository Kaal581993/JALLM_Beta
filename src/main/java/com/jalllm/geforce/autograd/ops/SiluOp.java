package com.jalllm.geforce.autograd.ops;

import com.jalllm.geforce.activation.SiLU;
import com.jalllm.geforce.autograd.Operation;
import com.jalllm.geforce.autograd.Variable;
import com.jalllm.geforce.tensor.Tensor;

import java.util.List;

/** SiLU (Swish) activation: z = SiLU(x) */
public final class SiluOp implements Operation {

    private final Variable x;
    private final Variable z;

    public SiluOp(Variable x) {
        this.x = x;
        Tensor result = SiLU.forward(x.value());
        this.z = Variable.fromOp(result, this, "silu");
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
        if (x.requiresGrad()) {
            Tensor gradX = SiLU.backward(x.value(), gradOutput);
            x.accumulateGradient(gradX);
        }
    }

    @Override
    public String name() {
        return "SiLU";
    }
}