package com.jalllm.geforce.training;

import com.jalllm.geforce.model.LanguageModel;
import com.jalllm.geforce.optimizer.Optimizer;

import java.io.Serializable;
import java.time.Instant;

/**
 * Training state for checkpointing and resuming.
 */
public final class TrainingState implements Serializable {

    private static final long serialVersionUID = 1L;

    private int epoch;
    private int step;
    private int globalStep;
    private float learningRate;
    private float trainLoss;
    private float valLoss;
    private float perplexity;
    private Instant timestamp;
    private String modelConfigJson;
    private int maxSteps;

    public TrainingState() {
        this.epoch = 0;
        this.step = 0;
        this.globalStep = 0;
        this.learningRate = 0.0f;
        this.trainLoss = Float.MAX_VALUE;
        this.valLoss = Float.MAX_VALUE;
        this.perplexity = Float.MAX_VALUE;
        this.timestamp = Instant.now();
        this.maxSteps = 0;
    }

    public float loss() {
        return trainLoss;
    }

    public int maxSteps() {
        return maxSteps;
    }

    public void setMaxSteps(int maxSteps) {
        this.maxSteps = maxSteps;
    }

    // Getters and setters
    public int epoch() { return epoch; }
    public void setEpoch(int epoch) { this.epoch = epoch; }

    public int step() { return step; }
    public void setStep(int step) { this.step = step; }

    public int globalStep() { return globalStep; }
    public void setGlobalStep(int globalStep) { this.globalStep = globalStep; }

    public float learningRate() { return learningRate; }
    public void setLearningRate(float learningRate) { this.learningRate = learningRate; }

    public float trainLoss() { return trainLoss; }
    public void setTrainLoss(float trainLoss) { this.trainLoss = trainLoss; }

    public float valLoss() { return valLoss; }
    public void setValLoss(float valLoss) { this.valLoss = valLoss; }

    public float perplexity() { return perplexity; }
    public void setPerplexity(float perplexity) { this.perplexity = perplexity; }

    public Instant timestamp() { return timestamp; }
    public void setTimestamp(Instant timestamp) { this.timestamp = timestamp; }

    public String modelConfigJson() { return modelConfigJson; }
    public void setModelConfigJson(String modelConfigJson) { this.modelConfigJson = modelConfigJson; }

    @Override
    public String toString() {
        return String.format("TrainingState(epoch=%d, step=%d, globalStep=%d, lr=%.6f, trainLoss=%.4f, valLoss=%.4f, ppl=%.2f)",
                epoch, step, globalStep, learningRate, trainLoss, valLoss, perplexity);
    }
}