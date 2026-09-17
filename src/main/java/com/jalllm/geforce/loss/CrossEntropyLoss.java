package com.jalllm.geforce.loss;

import com.jalllm.geforce.autograd.Variable;
import com.jalllm.geforce.autograd.VariableOps;
import com.jalllm.geforce.autograd.ops.SoftmaxOp;
import com.jalllm.geforce.tensor.Tensor;
import com.jalllm.geforce.tensor.TensorShape;

import java.util.Arrays;

/**
 * Cross-Entropy Loss for language modeling.
 * Combines LogSoftmax and NLLLoss in a numerically stable way.
 */
public final class CrossEntropyLoss {

    private final int ignoreIndex;
    private final boolean averageLoss;

    public CrossEntropyLoss() {
        this(-100, true);
    }

    public CrossEntropyLoss(int ignoreIndex) {
        this(ignoreIndex, true);
    }

    public CrossEntropyLoss(int ignoreIndex, boolean averageLoss) {
        this.ignoreIndex = ignoreIndex;
        this.averageLoss = averageLoss;
    }

    /**
     * Computes cross-entropy loss with autograd support.
     *
     * @param logits  Predicted logits [batch, seqLen, vocabSize] or [batch*seqLen, vocabSize]
     * @param targets Target token IDs [batch, seqLen] or [batch*seqLen]
     * @return Scalar loss variable
     */
    public Variable forward(Variable logits, Variable targets) {
        return new CrossEntropyOp(logits, targets, ignoreIndex, averageLoss).output();
    }

    /**
     * Computes cross-entropy loss without autograd (for evaluation).
     */
    public Tensor forward(Tensor logits, Tensor targets) {
        return forwardInference(logits, targets, ignoreIndex, averageLoss);
    }

    public static Tensor forwardInference(Tensor logits, Tensor targets) {
        return forwardInference(logits, targets, -100, true);
    }

    public static Tensor forwardInference(Tensor logits, Tensor targets, int ignoreIndex, boolean averageLoss) {
        int[] logitsShape = logits.shape();
        int[] targetsShape = targets.shape();

        // Flatten if needed
        int batchSeqLen;
        int vocabSize;
        if (logitsShape.length == 3) {
            batchSeqLen = logitsShape[0] * logitsShape[1];
            vocabSize = logitsShape[2];
        } else if (logitsShape.length == 2) {
            batchSeqLen = logitsShape[0];
            vocabSize = logitsShape[1];
        } else {
            throw new IllegalArgumentException("Logits must be 2D or 3D tensor");
        }

        if (targetsShape.length == 2) {
            if (targetsShape[0] * targetsShape[1] != batchSeqLen) {
                throw new IllegalArgumentException("Targets shape mismatch");
            }
        } else if (targetsShape.length == 1) {
            if (targetsShape[0] != batchSeqLen) {
                throw new IllegalArgumentException("Targets shape mismatch");
            }
        } else {
            throw new IllegalArgumentException("Targets must be 1D or 2D tensor");
        }

        float[] logitsData = logits.data();
        float[] targetsData = targets.data();

        float totalLoss = 0.0f;
        int validCount = 0;

        for (int i = 0; i < batchSeqLen; i++) {
            int target = (int) targetsData[i];
            // Check ignore index first, before range validation
            if (target == ignoreIndex) {
                continue;
            }

            // Only validate range if not ignored
            if (target < 0 || target >= vocabSize) {
                throw new IllegalArgumentException("Target " + target + " out of range [0, " + vocabSize + ")");
            }

            int offset = i * vocabSize;

            // LogSoftmax: log(softmax(x)) = x - log(sum(exp(x)))
            float maxLogit = -Float.MAX_VALUE;
            for (int v = 0; v < vocabSize; v++) {
                if (logitsData[offset + v] > maxLogit) {
                    maxLogit = logitsData[offset + v];
                }
            }

            float sumExp = 0.0f;
            for (int v = 0; v < vocabSize; v++) {
                sumExp += (float) Math.exp(logitsData[offset + v] - maxLogit);
            }

            float logSumExp = (float) Math.log(sumExp) + maxLogit;
            float logProb = logitsData[offset + target] - logSumExp;

            totalLoss -= logProb;
            validCount++;
        }

        float loss = averageLoss && validCount > 0 ? totalLoss / validCount : totalLoss;
        return Tensor.scalar(loss);
    }

    /**
     * Custom Operation for CrossEntropyLoss with autograd support.
     */
    private static final class CrossEntropyOp implements com.jalllm.geforce.autograd.Operation {

        private final Variable logits;
        private final Variable targets;
        private final int ignoreIndex;
        private final boolean averageLoss;
        private final Variable output;
        private final Variable logProbs; // Store for backward

        CrossEntropyOp(Variable logits, Variable targets, int ignoreIndex, boolean averageLoss) {
            this.logits = logits;
            this.targets = targets;
            this.ignoreIndex = ignoreIndex;
            this.averageLoss = averageLoss;

            Tensor lossTensor = forwardInference(logits.value(), targets.value(), ignoreIndex, averageLoss);
            this.output = Variable.fromOp(lossTensor, this, "cross_entropy_loss");

            // Compute and store log_probs for backward
            this.logProbs = computeLogProbs(logits.value());
        }

        @Override
        public java.util.List<Variable> inputs() {
            return java.util.List.of(logits, targets);
        }

        @Override
        public Variable output() {
            return output;
        }

        @Override
        public void backward(Tensor gradOutput) {
            // gradOutput is scalar (1.0 for loss)
            float gradScale = gradOutput.data()[0];

            int[] logitsShape = logits.value().shape();
            int batchSeqLen;
            int vocabSize;
            if (logitsShape.length == 3) {
                batchSeqLen = logitsShape[0] * logitsShape[1];
                vocabSize = logitsShape[2];
            } else {
                batchSeqLen = logitsShape[0];
                vocabSize = logitsShape[1];
            }

            float[] targetsData = targets.value().data();
            float[] logProbsData = logProbs.value().data();
            float[] gradLogitsData = new float[batchSeqLen * vocabSize];

            int validCount = 0;
            for (int i = 0; i < batchSeqLen; i++) {
                int target = (int) targetsData[i];
                // Check ignore index first, before range validation
                if (target == ignoreIndex) {
                    continue;
                }
                validCount++;

                int offset = i * vocabSize;
                // Gradient of cross-entropy: softmax - one_hot
                for (int v = 0; v < vocabSize; v++) {
                    float prob = (float) Math.exp(logProbsData[offset + v]);
                    float targetProb = (v == target) ? 1.0f : 0.0f;
                    gradLogitsData[offset + v] = (prob - targetProb) * gradScale;
                }
            }

            if (averageLoss && validCount > 0) {
                float scale = 1.0f / validCount;
                for (int i = 0; i < gradLogitsData.length; i++) {
                    gradLogitsData[i] *= scale;
                }
            }

            if (logits.requiresGrad()) {
                logits.accumulateGradient(new Tensor(logits.value().shape(), gradLogitsData));
            }
            // targets doesn't receive gradients
        }

        @Override
        public String name() {
            return "CrossEntropyLoss";
        }

        private Variable computeLogProbs(Tensor logits) {
            int[] shape = logits.shape();
            int batchSeqLen;
            int vocabSize;
            if (shape.length == 3) {
                batchSeqLen = shape[0] * shape[1];
                vocabSize = shape[2];
            } else {
                batchSeqLen = shape[0];
                vocabSize = shape[1];
            }

            float[] logitsData = logits.data();
            float[] logProbsData = new float[logitsData.length];

            for (int i = 0; i < batchSeqLen; i++) {
                int offset = i * vocabSize;

                float maxLogit = -Float.MAX_VALUE;
                for (int v = 0; v < vocabSize; v++) {
                    if (logitsData[offset + v] > maxLogit) {
                        maxLogit = logitsData[offset + v];
                    }
                }

                float sumExp = 0.0f;
                for (int v = 0; v < vocabSize; v++) {
                    sumExp += (float) Math.exp(logitsData[offset + v] - maxLogit);
                }

                float logSumExp = (float) Math.log(sumExp) + maxLogit;
                for (int v = 0; v < vocabSize; v++) {
                    logProbsData[offset + v] = logitsData[offset + v] - logSumExp;
                }
            }

            return Variable.fromOp(new Tensor(logits.shape(), logProbsData), null, "log_probs");
        }
    }
}