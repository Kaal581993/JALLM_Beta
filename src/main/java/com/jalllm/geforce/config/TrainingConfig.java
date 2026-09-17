package com.jalllm.geforce.config;

import java.io.Serializable;

/**
 * Training configuration.
 */
public final class TrainingConfig implements Serializable {

    private static final long serialVersionUID = 1L;

    private int batchSize = 4;
    private int maxSeqLen = 128;
    private int epochs = 10;
    private float learningRate = 3e-4f;
    private float weightDecay = 0.01f;
    private float beta1 = 0.9f;
    private float beta2 = 0.999f;
    private float eps = 1e-8f;
    private float gradClip = 1.0f;
    private int warmupSteps = 100;
    private int evalInterval = 100;
    private int saveInterval = 500;
    private int logInterval = 10;
    private String optimizer = "adamw"; // adamw, sgd
    private String lrSchedule = "cosine"; // constant, cosine, linear
    private float minLr = 1e-5f;
    private int seed = 42;
    private boolean mixedPrecision = false; // Not implemented yet
    private int numWorkers = 0; // Not used in pure Java

    public TrainingConfig() {}

    public TrainingConfig(int batchSize, int maxSeqLen, int epochs, float learningRate) {
        this.batchSize = batchSize;
        this.maxSeqLen = maxSeqLen;
        this.epochs = epochs;
        this.learningRate = learningRate;
    }

    public TrainingConfig(int maxSteps, int batchSize, float learningRate, float weightDecay, float gradClip, int warmupSteps, String outputDir) {
        this.epochs = 1; // Not used when maxSteps is specified
        this.batchSize = batchSize;
        this.learningRate = learningRate;
        this.weightDecay = weightDecay;
        this.gradClip = gradClip;
        this.warmupSteps = warmupSteps;
        // Note: outputDir is not stored in TrainingConfig, handled by Trainer
    }

    public int maxSteps() {
        // Calculate max steps based on epochs and dataset size, or return a default
        return epochs * 1000; // Placeholder
    }

    // Getters and setters
    public int batchSize() { return batchSize; }
    public void setBatchSize(int batchSize) { this.batchSize = batchSize; }

    public int maxSeqLen() { return maxSeqLen; }
    public void setMaxSeqLen(int maxSeqLen) { this.maxSeqLen = maxSeqLen; }

    public int epochs() { return epochs; }
    public void setEpochs(int epochs) { this.epochs = epochs; }

    public float learningRate() { return learningRate; }
    public void setLearningRate(float learningRate) { this.learningRate = learningRate; }

    public float weightDecay() { return weightDecay; }
    public void setWeightDecay(float weightDecay) { this.weightDecay = weightDecay; }

    public float beta1() { return beta1; }
    public void setBeta1(float beta1) { this.beta1 = beta1; }

    public float beta2() { return beta2; }
    public void setBeta2(float beta2) { this.beta2 = beta2; }

    public float eps() { return eps; }
    public void setEps(float eps) { this.eps = eps; }

    public float gradClip() { return gradClip; }
    public void setGradClip(float gradClip) { this.gradClip = gradClip; }

    public int warmupSteps() { return warmupSteps; }
    public void setWarmupSteps(int warmupSteps) { this.warmupSteps = warmupSteps; }

    public int evalInterval() { return evalInterval; }
    public void setEvalInterval(int evalInterval) { this.evalInterval = evalInterval; }

    public int saveInterval() { return saveInterval; }
    public void setSaveInterval(int saveInterval) { this.saveInterval = saveInterval; }

    public int logInterval() { return logInterval; }
    public void setLogInterval(int logInterval) { this.logInterval = logInterval; }

    public String optimizer() { return optimizer; }
    public void setOptimizer(String optimizer) { this.optimizer = optimizer; }

    public String lrSchedule() { return lrSchedule; }
    public void setLrSchedule(String lrSchedule) { this.lrSchedule = lrSchedule; }

    public float minLr() { return minLr; }
    public void setMinLr(float minLr) { this.minLr = minLr; }

    public int seed() { return seed; }
    public void setSeed(int seed) { this.seed = seed; }

    public boolean mixedPrecision() { return mixedPrecision; }
    public void setMixedPrecision(boolean mixedPrecision) { this.mixedPrecision = mixedPrecision; }

    public int numWorkers() { return numWorkers; }
    public void setNumWorkers(int numWorkers) { this.numWorkers = numWorkers; }

    @Override
    public String toString() {
        return String.format("TrainingConfig(batch=%d, seqLen=%d, epochs=%d, lr=%.2e, wd=%.2e, clip=%.1f)",
                batchSize, maxSeqLen, epochs, learningRate, weightDecay, gradClip);
    }
}