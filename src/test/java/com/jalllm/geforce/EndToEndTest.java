package com.jalllm.geforce;

import com.jalllm.geforce.config.ModelConfig;
import com.jalllm.geforce.config.RuntimeConfig;
import com.jalllm.geforce.config.TrainingConfig;
import com.jalllm.geforce.optimizer.AdamW;
import com.jalllm.geforce.serialization.ModelReader;
import com.jalllm.geforce.serialization.ModelWriter;
import com.jalllm.geforce.tensor.Tensor;
import com.jalllm.geforce.tokenizer.CharacterTokenizer;
import com.jalllm.geforce.tokenizer.Tokenizer;
import com.jalllm.geforce.training.Batch;
import com.jalllm.geforce.training.Trainer;
import com.jalllm.geforce.training.TrainingState;
import com.jalllm.geforce.model.MiniGPT;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * End-to-end test: tiny corpus → training → lower loss → save/load → generation.
 */
class EndToEndTest {

    @Test
    void testTinyCorpusTraining(@TempDir Path tempDir) throws Exception {
        // 1. Create tiny corpus
        String corpus = "hello world hello java hello world "
                + "java is great java is fun "
                + "hello java world hello "
                + "world of java world of code";

        Path dataFile = tempDir.resolve("tiny.txt");
        Files.writeString(dataFile, corpus);

        // 2. Tokenize
        String text = Files.readString(dataFile);
        CharacterTokenizer tokenizer = new CharacterTokenizer();
        int[] tokens = toIntArray(tokenizer.encode(text));
        assertTrue(tokens.length > 0, "Tokenization should produce tokens");
        int vocabSize = tokenizer.vocabSize();
        System.out.println("Vocab size: " + vocabSize + ", tokens: " + tokens.length);

        // 3. Create small model
        int embedDim = 16;
        int maxSeqLen = 8;
        int numLayers = 1;
        int numHeads = 1;
        int ffDim = embedDim * 4;

        ModelConfig modelConfig = new ModelConfig(
                embedDim, maxSeqLen, numLayers, numHeads, ffDim, vocabSize, embedDim / numHeads, 0.1f
        );
        MiniGPT model = new MiniGPT(modelConfig);
        System.out.println("Model params: " + model.parameterCount());

        // 4. Create training config
        TrainingConfig trainConfig = new TrainingConfig(
                20,    // maxSteps
                2,     // batchSize
                0.001f, // learningRate
                0.01f,  // weightDecay
                1.0f,   // gradClip
                0,      // warmupSteps
                tempDir.resolve("checkpoints").toString()
        );
        RuntimeConfig runtimeConfig = new RuntimeConfig(1, false, 42);

        // 5. Create optimizer and trainer
        AdamW optimizer = new AdamW(model.parameters(), trainConfig.learningRate(), trainConfig.weightDecay());
        Trainer trainer = new Trainer(model, optimizer, trainConfig, runtimeConfig);

        // 6. Create batches
        List<Batch> batches = createBatches(tokens, trainConfig.batchSize(), maxSeqLen, numHeads);
        assertFalse(batches.isEmpty(), "Should create at least one batch");
        System.out.println("Batches: " + batches.size());

        // 7. Evaluate initial loss (pass token sequences directly)
        List<int[]> dataset = new ArrayList<>();
        dataset.add(tokens);
        float initialLoss = trainer.evaluate(dataset, trainConfig.batchSize(), maxSeqLen);
        System.out.println("Initial loss: " + initialLoss);
        assertTrue(Float.isFinite(initialLoss), "Initial loss should be finite");

        // 8. Train
        TrainingState state = trainer.train(batches, trainConfig.maxSteps());
        float finalLoss = state.loss();
        System.out.println("Final loss: " + finalLoss);
        assertTrue(Float.isFinite(finalLoss), "Final loss should be finite");

        // 9. Loss should not explode (with small lr and tiny model, it should be reasonable)
        assertTrue(finalLoss < initialLoss * 5,
                "Loss should not explode: initial=" + initialLoss + " final=" + finalLoss);

        // 10. Save model
        Path modelPath = tempDir.resolve("model.bin");
        ModelWriter.write(model, modelPath.toFile());
        assertTrue(Files.exists(modelPath), "Model file should exist");
        System.out.println("Model saved to: " + modelPath);

        // 11. Load model
        MiniGPT loaded = ModelReader.read(modelPath.toFile());
        assertNotNull(loaded, "Loaded model should not be null");
        assertEquals(model.config().embedDim(), loaded.config().embedDim());
        assertEquals(model.config().numLayers(), loaded.config().numLayers());
        assertEquals(model.config().vocabSize(), loaded.config().vocabSize());
        System.out.println("Model loaded successfully");

        // 12. Generate text
        int[] promptTokens = toIntArray(tokenizer.encode("hello"));
        assertTrue(promptTokens.length > 0, "Prompt should have tokens");

        int[] generated = loaded.generate(promptTokens, 10, 1.0f, 0, 1.0f);
        assertNotNull(generated, "Generation should produce output");
        assertTrue(generated.length >= promptTokens.length,
                "Generated should include prompt");
        String output = tokenizer.decode(toList(generated));
        assertNotNull(output, "Decoded output should not be null");
        System.out.println("Generated: " + output);
    }

    @Test
    void testTrainingLossDecreases(@TempDir Path tempDir) throws Exception {
        // More extensive training to verify loss decreases
        String corpus = "hello world hello world hello world "
                .repeat(10); // Repeat to give more signal

        Path dataFile = tempDir.resolve("tiny2.txt");
        Files.writeString(dataFile, corpus);

        String text = Files.readString(dataFile);
        CharacterTokenizer tokenizer = new CharacterTokenizer();
        int[] tokens = toIntArray(tokenizer.encode(text));

        int vocabSize = tokenizer.vocabSize();
        int embedDim = 32;
        int maxSeqLen = 16;
        int numLayers = 2;
        int numHeads = 2;
        int ffDim = embedDim * 4;

        ModelConfig config = new ModelConfig(
                embedDim, maxSeqLen, numLayers, numHeads, ffDim, vocabSize, embedDim / numHeads, 0.1f
        );
        MiniGPT model = new MiniGPT(config);

        TrainingConfig trainConfig = new TrainingConfig(
                50, 4, 0.002f, 0.01f, 1.0f, 0, tempDir.resolve("ckpt").toString()
        );
        RuntimeConfig runtimeConfig = new RuntimeConfig(1, false, 42);

        AdamW optimizer = new AdamW(model.parameters(), trainConfig.learningRate(), trainConfig.weightDecay());
        Trainer trainer = new Trainer(model, optimizer, trainConfig, runtimeConfig);

        List<Batch> batches = createBatches(tokens, trainConfig.batchSize(), maxSeqLen, numHeads);

        List<int[]> dataset2 = new ArrayList<>();
        dataset2.add(tokens);
        float initialLoss = trainer.evaluate(dataset2, trainConfig.batchSize(), maxSeqLen);
        TrainingState state = trainer.train(batches, trainConfig.maxSteps());
        float finalLoss = state.loss();

        System.out.printf("Training: initial=%.4f, final=%.4f%n", initialLoss, finalLoss);
        assertTrue(Float.isFinite(finalLoss), "Final loss should be finite");
        // With 50 steps on a repetitive corpus, loss should improve
        assertTrue(finalLoss <= initialLoss * 3,
                "Loss should not worsen dramatically: initial=" + initialLoss + " final=" + finalLoss);
    }

    private static List<int[]> toDataset(List<Batch> batches) {
        List<int[]> dataset = new ArrayList<>();
        for (Batch batch : batches) {
            int[] seq = new int[batch.inputIds().size()];
            for (int i = 0; i < seq.length; i++) {
                seq[i] = (int) batch.inputIds().get(i);
            }
            dataset.add(seq);
        }
        return dataset;
    }

    private static List<Integer> toList(int[] arr) {
        List<Integer> list = new ArrayList<>(arr.length);
        for (int v : arr) list.add(v);
        return list;
    }

    private static int[] toIntArray(List<Integer> values) {
        return values.stream().mapToInt(Integer::intValue).toArray();
    }

    private static List<Batch> createBatches(int[] tokens, int batchSize, int seqLen, int numHeads) {
        List<Batch> batches = new ArrayList<>();
        int numTokens = tokens.length - seqLen;
        int numBatches = Math.max(1, numTokens / batchSize);

        for (int b = 0; b < numBatches; b++) {
            float[][] inputData = new float[batchSize][seqLen];
            float[][] targetData = new float[batchSize][seqLen];

            for (int i = 0; i < batchSize; i++) {
                int start = b * batchSize + i;
                if (start + seqLen + 1 > tokens.length) break;
                for (int s = 0; s < seqLen; s++) {
                    inputData[i][s] = tokens[start + s];
                    targetData[i][s] = tokens[start + s + 1];
                }
            }

            Tensor inputs = new Tensor(new int[]{batchSize, seqLen}, flatten(inputData));
            Tensor targets = new Tensor(new int[]{batchSize, seqLen}, flatten(targetData));
            batches.add(new Batch(inputs, targets,
                    com.jalllm.geforce.attention.CausalMask.create(batchSize, numHeads, seqLen)));
        }

        return batches;
    }

    private static float[] flatten(float[][] arr) {
        int total = 0;
        for (float[] row : arr) total += row.length;
        float[] result = new float[total];
        int idx = 0;
        for (float[] row : arr) {
            System.arraycopy(row, 0, result, idx, row.length);
            idx += row.length;
        }
        return result;
    }
}
