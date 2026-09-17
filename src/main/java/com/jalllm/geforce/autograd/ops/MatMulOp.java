package com.jalllm.geforce.autograd.ops;

import com.jalllm.geforce.autograd.Operation;
import com.jalllm.geforce.autograd.Variable;
import com.jalllm.geforce.math.MatrixMath;
import com.jalllm.geforce.tensor.Tensor;

import java.util.List;

/** Matrix multiplication: z = x @ y */
public final class MatMulOp implements Operation {

    private final Variable x;
    private final Variable y;
    private final Variable z;

    public MatMulOp(Variable x, Variable y) {
        this.x = x;
        this.y = y;
        Tensor result = MatrixMath.matmul(x.value(), y.value());
        this.z = Variable.fromOp(result, this, "matmul");
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
        // z = x @ y
        // grad_x = grad_z @ y^T
        // grad_y = x^T @ grad_z
        if (x.requiresGrad()) {
            Tensor yT = MatrixMath.transpose(y.value());
            Tensor gradX = MatrixMath.matmul(gradOutput, yT);
            x.accumulateGradient(gradX.sumToShape(x.value().shape()));
        }
        if (y.requiresGrad()) {
            Tensor xT = MatrixMath.transpose(x.value());
            Tensor gradY = MatrixMath.matmul(xT, gradOutput);
            y.accumulateGradient(gradY.sumToShape(y.value().shape()));
        }
    }

    @Override
    public String name() {
        return "MatMul";
    }
}
