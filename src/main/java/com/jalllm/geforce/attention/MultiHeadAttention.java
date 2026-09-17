package com.jalllm.geforce.attention;

import com.jalllm.geforce.autograd.Variable;
import com.jalllm.geforce.autograd.VariableOps;
import com.jalllm.geforce.math.MatrixMath;
import com.jalllm.geforce.math.RandomInitializer;
import com.jalllm.geforce.tensor.Tensor;
import com.jalllm.geforce.tensor.TensorShape;

import java.util.ArrayList;
import java.util.List;

/**
 * Multi-Head Attention.
 * Projects Q, K, V into multiple heads, applies attention, then concatenates and projects output.
 */
public final class MultiHeadAttention {

    private final int embedDim;
    private final int numHeads;
    private final int headDim;
    private final Variable wQ;
    private final Variable wK;
    private final Variable wV;
    private final Variable wO;
    private final Variable bQ;
    private final Variable bK;
    private final Variable bV;
    private final Variable bO;

    public MultiHeadAttention(int embedDim, int numHeads) {
        if (embedDim % numHeads != 0) {
            throw new IllegalArgumentException("embedDim (" + embedDim + ") must be divisible by numHeads (" + numHeads + ")");
        }
        this.embedDim = embedDim;
        this.numHeads = numHeads;
        this.headDim = embedDim / numHeads;

        // Q, K, V projections: [embedDim, embedDim]
        this.wQ = Variable.of(new Tensor(new int[]{embedDim, embedDim}), "mha_wq");
        this.wK = Variable.of(new Tensor(new int[]{embedDim, embedDim}), "mha_wk");
        this.wV = Variable.of(new Tensor(new int[]{embedDim, embedDim}), "mha_wv");
        this.wO = Variable.of(new Tensor(new int[]{embedDim, embedDim}), "mha_wo");

        // Biases: [embedDim]
        this.bQ = Variable.of(new Tensor(new int[]{embedDim}), "mha_bq");
        this.bK = Variable.of(new Tensor(new int[]{embedDim}), "mha_bk");
        this.bV = Variable.of(new Tensor(new int[]{embedDim}), "mha_bv");
        this.bO = Variable.of(new Tensor(new int[]{embedDim}), "mha_bo");

        // Initialize with Xavier
        RandomInitializer.xavierUniform(wQ.value(), embedDim, embedDim);
        RandomInitializer.xavierUniform(wK.value(), embedDim, embedDim);
        RandomInitializer.xavierUniform(wV.value(), embedDim, embedDim);
        RandomInitializer.xavierUniform(wO.value(), embedDim, embedDim);
        RandomInitializer.constant(bQ.value(), 0.0f);
        RandomInitializer.constant(bK.value(), 0.0f);
        RandomInitializer.constant(bV.value(), 0.0f);
        RandomInitializer.constant(bO.value(), 0.0f);
    }

    /**
     * Forward pass with autograd support.
     * Input: [batch, seqLen, embedDim]
     * Output: [batch, seqLen, embedDim]
     */
    public Variable forward(Variable x, Variable mask) {
        return forward(x, mask, true);
    }

    /**
     * Forward pass without autograd (for inference).
     */
    public Tensor forward(Tensor x, Tensor mask) {
        return forwardInference(x, mask);
    }

    private Variable forward(Variable x, Variable mask, boolean trackGrad) {
        if (!trackGrad) {
            return Variable.of(forwardInference(x.value(), mask != null ? mask.value() : null), "mha_out");
        }

        // Linear projections
        Variable q = VariableOps.add(VariableOps.matmul(x, wQ), bQ);
        Variable k = VariableOps.add(VariableOps.matmul(x, wK), bK);
        Variable v = VariableOps.add(VariableOps.matmul(x, wV), bV);

        // Reshape for multi-head: [batch, seqLen, embedDim] -> [batch, seqLen, numHeads, headDim] -> [batch, numHeads, seqLen, headDim]
        int[] xShape = x.value().shape();
        int batch = xShape[0];
        int seqLen = xShape[1];

        q = VariableOps.reshape(q, batch, seqLen, numHeads, headDim);
        k = VariableOps.reshape(k, batch, seqLen, numHeads, headDim);
        v = VariableOps.reshape(v, batch, seqLen, numHeads, headDim);

        // Transpose to [batch, numHeads, seqLen, headDim]
        q = transposeForAttention(q);
        k = transposeForAttention(k);
        v = transposeForAttention(v);

        // Reshape mask if provided
        Variable attnMask = mask;
        if (mask != null) {
            // Mask shape: [batch, seqLen, seqLen] or [batch, 1, seqLen, seqLen]
            // Need to expand to [batch, numHeads, seqLen, seqLen]
            int[] maskShape = mask.value().shape();
            if (maskShape.length == 3) {
                // [batch, seqLen, seqLen] -> [batch, 1, seqLen, seqLen] -> expand to heads
                attnMask = expandMask(mask, numHeads);
            }
        }

        // Scaled dot-product attention
        Variable attnOut = ScaledDotProductAttention.forward(q, k, v, attnMask);

        // Transpose back: [batch, numHeads, seqLen, headDim] -> [batch, seqLen, numHeads, headDim]
        attnOut = transposeFromAttention(attnOut);

        // Reshape: [batch, seqLen, numHeads, headDim] -> [batch, seqLen, embedDim]
        attnOut = VariableOps.reshape(attnOut, batch, seqLen, embedDim);

        // Output projection
        Variable out = VariableOps.add(VariableOps.matmul(attnOut, wO), bO);

        return out;
    }

    private Tensor forwardInference(Tensor x, Tensor mask) {
        int[] xShape = x.shape();
        int batch = xShape[0];
        int seqLen = xShape[1];

        // Linear projections
        Tensor q = MatrixMath.matmul(x, wQ.value()).add(bQ.value());
        Tensor k = MatrixMath.matmul(x, wK.value()).add(bK.value());
        Tensor v = MatrixMath.matmul(x, wV.value()).add(bV.value());

        // Reshape for multi-head
        q = q.reshape(batch, seqLen, numHeads, headDim);
        k = k.reshape(batch, seqLen, numHeads, headDim);
        v = v.reshape(batch, seqLen, numHeads, headDim);

        // Transpose to [batch, numHeads, seqLen, headDim]
        q = transposeForAttentionInference(q);
        k = transposeForAttentionInference(k);
        v = transposeForAttentionInference(v);

        // Reshape mask if provided
        Tensor attnMask = mask;
        if (mask != null) {
            int[] maskShape = mask.shape();
            if (maskShape.length == 3) {
                attnMask = expandMaskInference(mask, numHeads);
            }
        }

        // Scaled dot-product attention
        Tensor attnOut = ScaledDotProductAttention.forward(q, k, v, attnMask);

        // Transpose back
        attnOut = transposeFromAttentionInference(attnOut);

        // Reshape
        attnOut = attnOut.reshape(batch, seqLen, embedDim);

        // Output projection
        Tensor out = MatrixMath.matmul(attnOut, wO.value()).add(bO.value());

        return out;
    }

    private Variable transposeForAttention(Variable x) {
        // [batch, seqLen, numHeads, headDim] -> [batch, numHeads, seqLen, headDim]
        // This is a permutation of dimensions 1 and 2
        Transpose12Op op = new Transpose12Op(x);
        return op.output();
    }

    private Variable transposeFromAttention(Variable x) {
        // [batch, numHeads, seqLen, headDim] -> [batch, seqLen, numHeads, headDim]
        Transpose12Op op = new Transpose12Op(x);
        return op.output();
    }

    private Tensor transposeForAttentionInference(Tensor x) {
        // [batch, seqLen, numHeads, headDim] -> [batch, numHeads, seqLen, headDim]
        int[] shape = x.shape();
        int batch = shape[0];
        int seqLen = shape[1];
        int numHeads = shape[2];
        int headDim = shape[3];

        float[] inData = x.data();
        float[] outData = new float[inData.length];

        for (int b = 0; b < batch; b++) {
            for (int h = 0; h < numHeads; h++) {
                for (int s = 0; s < seqLen; s++) {
                    for (int d = 0; d < headDim; d++) {
                        int inIdx = ((b * seqLen + s) * numHeads + h) * headDim + d;
                        int outIdx = ((b * numHeads + h) * seqLen + s) * headDim + d;
                        outData[outIdx] = inData[inIdx];
                    }
                }
            }
        }

        return new Tensor(new int[]{batch, numHeads, seqLen, headDim}, outData);
    }

    private Tensor transposeFromAttentionInference(Tensor x) {
        // [batch, numHeads, seqLen, headDim] -> [batch, seqLen, numHeads, headDim]
        int[] shape = x.shape();
        int batch = shape[0];
        int numHeads = shape[1];
        int seqLen = shape[2];
        int headDim = shape[3];

        float[] inData = x.data();
        float[] outData = new float[inData.length];

        for (int b = 0; b < batch; b++) {
            for (int s = 0; s < seqLen; s++) {
                for (int h = 0; h < numHeads; h++) {
                    for (int d = 0; d < headDim; d++) {
                        int inIdx = ((b * numHeads + h) * seqLen + s) * headDim + d;
                        int outIdx = ((b * seqLen + s) * numHeads + h) * headDim + d;
                        outData[outIdx] = inData[inIdx];
                    }
                }
            }
        }

        return new Tensor(new int[]{batch, seqLen, numHeads, headDim}, outData);
    }

    private Variable expandMask(Variable mask, int numHeads) {
        // [batch, seqLen, seqLen] -> [batch, numHeads, seqLen, seqLen]
        ExpandMaskOp op = new ExpandMaskOp(mask, numHeads);
        return op.output();
    }

    private Tensor expandMaskInference(Tensor mask, int numHeads) {
        int[] shape = mask.shape();
        int batch = shape[0];
        int seqLen = shape[1];

        float[] inData = mask.data();
        float[] outData = new float[batch * numHeads * seqLen * seqLen];

        for (int b = 0; b < batch; b++) {
            for (int h = 0; h < numHeads; h++) {
                for (int i = 0; i < seqLen; i++) {
                    for (int j = 0; j < seqLen; j++) {
                        int inIdx = (b * seqLen + i) * seqLen + j;
                        int outIdx = ((b * numHeads + h) * seqLen + i) * seqLen + j;
                        outData[outIdx] = inData[inIdx];
                    }
                }
            }
        }

        return new Tensor(new int[]{batch, numHeads, seqLen, seqLen}, outData);
    }

    // Getters for parameters
    public Variable wQ() { return wQ; }
    public Variable wK() { return wK; }
    public Variable wV() { return wV; }
    public Variable wO() { return wO; }
    public Variable bQ() { return bQ; }
    public Variable bK() { return bK; }
    public Variable bV() { return bV; }
    public Variable bO() { return bO; }

    public int embedDim() { return embedDim; }
    public int numHeads() { return numHeads; }
    public int headDim() { return headDim; }

    public List<Variable> parameters() {
        List<Variable> params = new ArrayList<>();
        params.add(wQ); params.add(wK); params.add(wV); params.add(wO);
        params.add(bQ); params.add(bK); params.add(bV); params.add(bO);
        return params;
    }

    // Custom operations for transpose and mask expansion
    private static final class Transpose12Op implements com.jalllm.geforce.autograd.Operation {
        private final Variable x;
        private final Variable output;

        Transpose12Op(Variable x) {
            this.x = x;
            // [batch, seqLen, numHeads, headDim] <-> [batch, numHeads, seqLen, headDim]
            int[] shape = x.value().shape();
            int batch = shape[0];
            int dim1 = shape[1];
            int dim2 = shape[2];
            int dim3 = shape[3];

            float[] inData = x.value().data();
            float[] outData = new float[inData.length];

            for (int b = 0; b < batch; b++) {
                for (int i = 0; i < dim1; i++) {
                    for (int j = 0; j < dim2; j++) {
                        for (int k = 0; k < dim3; k++) {
                            int inIdx = ((b * dim1 + i) * dim2 + j) * dim3 + k;
                            int outIdx = ((b * dim2 + j) * dim1 + i) * dim3 + k;
                            outData[outIdx] = inData[inIdx];
                        }
                    }
                }
            }

            int[] outShape = new int[]{batch, dim2, dim1, dim3};
            this.output = Variable.fromOp(new Tensor(outShape, outData), this, "transpose12");
        }

        @Override
        public List<Variable> inputs() {
            return List.of(x);
        }

        @Override
        public Variable output() {
            return output;
        }

        @Override
        public void backward(Tensor gradOutput) {
            // Transpose gradient back
            int[] shape = gradOutput.shape();
            int batch = shape[0];
            int dim1 = shape[1];
            int dim2 = shape[2];
            int dim3 = shape[3];

            float[] gradOutData = gradOutput.data();
            float[] gradInData = new float[gradOutData.length];

            for (int b = 0; b < batch; b++) {
                for (int i = 0; i < dim1; i++) {
                    for (int j = 0; j < dim2; j++) {
                        for (int k = 0; k < dim3; k++) {
                            int outIdx = ((b * dim1 + i) * dim2 + j) * dim3 + k;
                            int inIdx = ((b * dim2 + j) * dim1 + i) * dim3 + k;
                            gradInData[inIdx] = gradOutData[outIdx];
                        }
                    }
                }
            }

            int[] inShape = new int[]{batch, dim2, dim1, dim3};
            if (x.requiresGrad()) {
                x.accumulateGradient(new Tensor(inShape, gradInData));
            }
        }

        @Override
        public String name() {
            return "Transpose12";
        }
    }

    private static final class ExpandMaskOp implements com.jalllm.geforce.autograd.Operation {
        private final Variable mask;
        private final int numHeads;
        private final Variable output;

        ExpandMaskOp(Variable mask, int numHeads) {
            this.mask = mask;
            this.numHeads = numHeads;

            int[] shape = mask.value().shape();
            int batch = shape[0];
            int seqLen = shape[1];

            float[] inData = mask.value().data();
            float[] outData = new float[batch * numHeads * seqLen * seqLen];

            for (int b = 0; b < batch; b++) {
                for (int h = 0; h < numHeads; h++) {
                    for (int i = 0; i < seqLen; i++) {
                        for (int j = 0; j < seqLen; j++) {
                            int inIdx = (b * seqLen + i) * seqLen + j;
                            int outIdx = ((b * numHeads + h) * seqLen + i) * seqLen + j;
                            outData[outIdx] = inData[inIdx];
                        }
                    }
                }
            }

            this.output = Variable.fromOp(new Tensor(new int[]{batch, numHeads, seqLen, seqLen}, outData), this, "expand_mask");
        }

        @Override
        public List<Variable> inputs() {
            return List.of(mask);
        }

        @Override
        public Variable output() {
            return output;
        }

        @Override
        public void backward(Tensor gradOutput) {
            // Sum gradients over heads
            int[] shape = gradOutput.shape();
            int batch = shape[0];
            int numHeads = shape[1];
            int seqLen = shape[2];

            float[] gradOutData = gradOutput.data();
            float[] gradInData = new float[batch * seqLen * seqLen];

            for (int b = 0; b < batch; b++) {
                for (int i = 0; i < seqLen; i++) {
                    for (int j = 0; j < seqLen; j++) {
                        float sum = 0;
                        for (int h = 0; h < numHeads; h++) {
                            int outIdx = ((b * numHeads + h) * seqLen + i) * seqLen + j;
                            sum += gradOutData[outIdx];
                        }
                        gradInData[(b * seqLen + i) * seqLen + j] = sum;
                    }
                }
            }

            if (mask.requiresGrad()) {
                mask.accumulateGradient(new Tensor(new int[]{batch, seqLen, seqLen}, gradInData));
            }
        }

        @Override
        public String name() {
            return "ExpandMask";
        }
    }
}