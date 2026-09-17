package com.jalllm.geforce.training;

import com.jalllm.geforce.autograd.Variable;
import com.jalllm.geforce.autograd.ComputationalGraph;
import com.jalllm.geforce.config.RuntimeConfig;
import com.jalllm.geforce.config.TrainingConfig;
import com.jalllm.geforce.loss.CrossEntropyLoss;
import com.jalllm.geforce.model.LanguageModel;
import com.jalllm.geforce.optimizer.Optimizer;
import com.jalllm.geforce.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.function.Consumer;

/**
 * Training loop for language models.
 */
public final class Trainer {

    private final LanguageModel model;
    private final Optimizer optimizer;
    private final CrossEntropyLoss lossFn;
    private final TrainingState state;
    private final Random random;

    private Consumer<TrainingState> onStepEnd;
    private Consumer<TrainingState> onEpochEnd;

    public Trainer(LanguageModel model, Optimizer optimizer, CrossEntropyLoss lossFn) {
        this.model = model;
        this.optimizer = optimizer;
        this.lossFn = lossFn;
        this.state = new TrainingState();
        this.random = new Random();
    }

    public Trainer(LanguageModel model, Optimizer optimizer) {
        this(model, optimizer, new CrossEntropyLoss());
    }

    public Trainer(LanguageModel model, Optimizer optimizer, TrainingConfig trainConfig, RuntimeConfig runtimeConfig) {
        this(model, optimizer, new CrossEntropyLoss());
        // TrainingConfig and RuntimeConfig are stored in state or used for configuration
        if (trainConfig != null) {
            state.setMaxSteps(trainConfig.maxSteps());
        }
    }

    /**
     * Trains for one epoch over the dataset.
     *
     * @param dataset List of token ID sequences
     * @param batchSize Batch size
     * @param maxSeqLen Maximum sequence length
     * @return Average loss for the epoch
     */
    public float trainEpoch(List<int[]> dataset, int batchSize, int maxSeqLen) {
        // Shuffle dataset
        List<int[]> shuffled = new ArrayList<>(dataset);
        for (int i = shuffled.size() - 1; i > 0; i--) {
            int j = random.nextInt(i + 1);
            int[] temp = shuffled.get(i);
            shuffled.set(i, shuffled.get(j));
            shuffled.set(j, temp);
        }

        float totalLoss = 0.0f;
        int numBatches = 0;

        for (int i = 0; i < shuffled.size(); i += batchSize) {
            int end = Math.min(i + batchSize, shuffled.size());
            int actualBatchSize = end - i;

            // Prepare batch
            Batch batch = createBatch(shuffled, i, end, maxSeqLen);

            // Forward pass
            Variable inputVar = Variable.of(batch.inputIds(), "input");
            Variable targetVar = Variable.of(batch.targetIds(), "target");
            Variable maskVar = batch.mask() != null ? Variable.of(batch.mask(), "mask") : null;

            Variable logits = model.forward(inputVar, maskVar);
            Variable loss = lossFn.forward(logits, targetVar);

            // Backward pass
            optimizer.zeroGrad();
            ComputationalGraph.backward(loss);
            optimizer.step();

            float lossVal = loss.value().data()[0];
            totalLoss += lossVal;
            numBatches++;

            // Update state
            state.setStep(state.step() + 1);
            state.setGlobalStep(state.globalStep() + 1);
            state.setTrainLoss(lossVal);
            state.setLearningRate(optimizer.learningRate());
            state.setTimestamp(java.time.Instant.now());

            // Callback
            if (onStepEnd != null) {
                onStepEnd.accept(state);
            }
        }

        state.setEpoch(state.epoch() + 1);
        if (onEpochEnd != null) {
            onEpochEnd.accept(state);
        }

        return numBatches > 0 ? totalLoss / numBatches : Float.MAX_VALUE;
    }

    /**
     * Evaluates the model on a dataset.
     */
    public float evaluate(List<int[]> dataset, int batchSize, int maxSeqLen) {
        float totalLoss = 0.0f;
        int numBatches = 0;

        for (int i = 0; i < dataset.size(); i += batchSize) {
            int end = Math.min(i + batchSize, dataset.size());
            Batch batch = createBatch(dataset, i, end, maxSeqLen);

            // Forward pass without autograd
            Tensor logits = model.forward(batch.inputIds(), batch.mask());
            Tensor loss = lossFn.forward(logits, batch.targetIds());

            totalLoss += loss.data()[0];
            numBatches++;
        }

        float avgLoss = numBatches > 0 ? totalLoss / numBatches : Float.MAX_VALUE;
        state.setValLoss(avgLoss);
        state.setPerplexity((float) Math.exp(Math.min(avgLoss, 20))); // Cap for numerical stability

        return avgLoss;
    }

    /**
     * Creates a batch from a slice of the dataset.
     */
    private Batch createBatch(List<int[]> dataset, int start, int end, int maxSeqLen) {
        int batchSize = end - start;
        int seqLen = maxSeqLen;

        float[] inputData = new float[batchSize * seqLen];
        float[] targetData = new float[batchSize * seqLen];

        for (int b = 0; b < batchSize; b++) {
            int[] sequence = dataset.get(start + b);
            int copyLen = Math.min(sequence.length - 1, seqLen);

            for (int s = 0; s < copyLen; s++) {
                inputData[b * seqLen + s] = sequence[s];
                targetData[b * seqLen + s] = sequence[s + 1];
            }
            // Pad remaining with pad token (0)
            for (int s = copyLen; s < seqLen; s++) {
                inputData[b * seqLen + s] = 0;
                targetData[b * seqLen + s] = -100; // ignore index
            }
        }

        Tensor inputIds = new Tensor(new int[]{batchSize, seqLen}, inputData);
        Tensor targetIds = new Tensor(new int[]{batchSize, seqLen}, targetData);
        Tensor mask = model instanceof com.jalllm.geforce.model.MiniGPT
                ? ((com.jalllm.geforce.model.MiniGPT) model).createCausalMask(batchSize, seqLen)
                : null;

        return new Batch(inputIds, targetIds, mask);
    }

    public TrainingState state() { return state; }
    public LanguageModel model() { return model; }
    public Optimizer optimizer() { return optimizer; }

    public void setOnStepEnd(Consumer<TrainingState> callback) { this.onStepEnd = callback; }
    public void setOnEpochEnd(Consumer<TrainingState> callback) { this.onEpochEnd = callback; }

    /**
     * Trains on a list of batches for a specified number of steps.
     */
    public TrainingState train(List<Batch> batches, int maxSteps) {
        int step = 0;
        int batchIndex = 0;

        while (step < maxSteps && batchIndex < batches.size()) {
            Batch batch = batches.get(batchIndex);

            // Forward pass
            Variable inputVar = Variable.of(batch.inputIds(), "input");
            Variable targetVar = Variable.of(batch.targetIds(), "target");
            Variable maskVar = batch.mask() != null ? Variable.of(batch.mask(), "mask") : null;

            Variable logits = model.forward(inputVar, maskVar);
            Variable loss = lossFn.forward(logits, targetVar);

            // Backward pass
            optimizer.zeroGrad();
            ComputationalGraph.backward(loss);
            optimizer.step();

            float lossVal = loss.value().data()[0];

            // Update state
            state.setStep(state.step() + 1);
            state.setGlobalStep(state.globalStep() + 1);
            state.setTrainLoss(lossVal);
            state.setLearningRate(optimizer.learningRate());
            state.setTimestamp(java.time.Instant.now());

            // Callback
            if (onStepEnd != null) {
                onStepEnd.accept(state);
            }

            step++;
            batchIndex++;
            if (batchIndex >= batches.size()) {
                batchIndex = 0; // Loop back to start of batches
            }
        }

        state.setEpoch(state.epoch() + 1);
        if (onEpochEnd != null) {
            onEpochEnd.accept(state);
        }

        return state;
    }
}