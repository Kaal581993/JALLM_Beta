package com.jalllm.geforce.autograd;

import com.jalllm.geforce.autograd.ops.*;
import com.jalllm.geforce.tensor.Tensor;

/**
 * Extension methods for Variable to make autograd operations more ergonomic.
 * Usage: Variable z = x.add(y); Variable z = x.matmul(y); etc.
 */
public final class VariableOps {

    private VariableOps() {}

    // Element-wise operations

    public static Variable add(Variable x, Variable y) {
        return new AddOp(x, y).output();
    }

    public static Variable add(Variable x, float scalar) {
        Variable s = Variable.constant(Tensor.scalar(scalar), "scalar");
        return new AddOp(x, s).output();
    }

    public static Variable sub(Variable x, Variable y) {
        return new SubOp(x, y).output();
    }

    public static Variable sub(Variable x, float scalar) {
        Variable s = Variable.constant(Tensor.scalar(scalar), "scalar");
        return new SubOp(x, s).output();
    }

    public static Variable mul(Variable x, Variable y) {
        return new MulOp(x, y).output();
    }

    public static Variable mul(Variable x, float scalar) {
        Variable s = Variable.constant(Tensor.scalar(scalar), "scalar");
        return new MulOp(x, s).output();
    }

    public static Variable div(Variable x, Variable y) {
        return new DivOp(x, y).output();
    }

    public static Variable div(Variable x, float scalar) {
        Variable s = Variable.constant(Tensor.scalar(scalar), "scalar");
        return new DivOp(x, s).output();
    }

    public static Variable neg(Variable x) {
        return new NegOp(x).output();
    }

    // Matrix multiplication

    public static Variable matmul(Variable x, Variable y) {
        return new MatMulOp(x, y).output();
    }

    // Activations

    public static Variable gelu(Variable x) {
        return new GeluOp(x).output();
    }

    public static Variable silu(Variable x) {
        return new SiluOp(x).output();
    }

    public static Variable softmax(Variable x) {
        return new SoftmaxOp(x).output();
    }

    // Reshape (no gradient needed for reshape itself, but we need to track it)
    public static Variable reshape(Variable x, int... shape) {
        return new ReshapeOp(x, shape).output();
    }

    /** Internal operation for reshape */
    private static final class ReshapeOp implements Operation {
        private final Variable x;
        private final Variable z;
        private final int[] originalShape;

        ReshapeOp(Variable x, int[] newShape) {
            this.x = x;
            this.originalShape = x.value().shape();
            this.z = Variable.fromOp(x.value().reshape(newShape), this, "reshape");
        }

        @Override
        public java.util.List<Variable> inputs() {
            return java.util.List.of(x);
        }

        @Override
        public Variable output() {
            return z;
        }

        @Override
        public void backward(Tensor gradOutput) {
            if (x.requiresGrad()) {
                // Reshape gradient back to original shape
                Tensor gradX = gradOutput.reshape(originalShape);
                x.accumulateGradient(gradX);
            }
        }

        @Override
        public String name() {
            return "Reshape";
        }
    }

    // Transpose (for 2D matrices)
    public static Variable transpose(Variable x) {
        return new TransposeOp(x).output();
    }

    private static final class TransposeOp implements Operation {
        private final Variable x;
        private final Variable z;

        TransposeOp(Variable x) {
            this.x = x;
            this.z = Variable.fromOp(com.jalllm.geforce.math.MatrixMath.transpose(x.value()), this, "transpose");
        }

        @Override
        public java.util.List<Variable> inputs() {
            return java.util.List.of(x);
        }

        @Override
        public Variable output() {
            return z;
        }

        @Override
        public void backward(Tensor gradOutput) {
            if (x.requiresGrad()) {
                // Transpose gradient back
                Tensor gradX = com.jalllm.geforce.math.MatrixMath.transpose(gradOutput);
                x.accumulateGradient(gradX);
            }
        }

        @Override
        public String name() {
            return "Transpose";
        }
    }

    // Sum reduction (for loss computation)
    public static Variable sum(Variable x) {
        return new SumOp(x).output();
    }

    private static final class SumOp implements Operation {
        private final Variable x;
        private final Variable z;

        SumOp(Variable x) {
            this.x = x;
            float sum = 0.0f;
            for (float v : x.value().data()) {
                sum += v;
            }
            this.z = Variable.fromOp(Tensor.scalar(sum), this, "sum");
        }

        @Override
        public java.util.List<Variable> inputs() {
            return java.util.List.of(x);
        }

        @Override
        public Variable output() {
            return z;
        }

        @Override
        public void backward(Tensor gradOutput) {
            if (x.requiresGrad()) {
                // Gradient of sum is broadcast to all elements
                float gradVal = gradOutput.data()[0];
                Tensor gradX = Tensor.zeros(x.value().shape());
                gradX.fill(gradVal);
                x.accumulateGradient(gradX);
            }
        }

        @Override
        public String name() {
            return "Sum";
        }
    }

    // Mean reduction
    public static Variable mean(Variable x) {
        return new MeanOp(x).output();
    }

    private static final class MeanOp implements Operation {
        private final Variable x;
        private final Variable z;

        MeanOp(Variable x) {
            this.x = x;
            float sum = 0.0f;
            for (float v : x.value().data()) {
                sum += v;
            }
            this.z = Variable.fromOp(Tensor.scalar(sum / x.value().elementCount()), this, "mean");
        }

        @Override
        public java.util.List<Variable> inputs() {
            return java.util.List.of(x);
        }

        @Override
        public Variable output() {
            return z;
        }

        @Override
        public void backward(Tensor gradOutput) {
            if (x.requiresGrad()) {
                float gradVal = gradOutput.data()[0] / x.value().elementCount();
                Tensor gradX = Tensor.zeros(x.value().shape());
                gradX.fill(gradVal);
                x.accumulateGradient(gradX);
            }
        }

        @Override
        public String name() {
            return "Mean";
        }
    }
}
