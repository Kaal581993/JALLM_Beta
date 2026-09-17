package com.jalllm.geforce.autograd.ops;

import com.jalllm.geforce.activation.Softmax;
import com.jalllm.geforce.autograd.Operation;
import com.jalllm.geforce.autograd.Variable;
import com.jalllm.geforce.tensor.Tensor;

import java.util.List;

/** Softmax activation: z = Softmax(x) along last dimension */
public final class SoftmaxOp implements Operation {

    private final Variable x;
    private final Variable z;

    public SoftmaxOp(Variable x) {
        this.x = x;
        Tensor result = Softmax.forward(x.value());
        this.z = Variable.fromOp(result, this, "softmax");
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
            Tensor gradX = Softmax.backward(z.value(), gradOutput);
            x.accumulateGradient(gradX);
        }
    }

    @Override
    public String name() {
        return "Softmax";
    }
}