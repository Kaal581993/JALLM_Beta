package com.jalllm.geforce;

import com.jalllm.geforce.activation.Softmax;

import com.jalllm.geforce.benchmark.InferenceBenchmark;
import com.jalllm.geforce.benchmark.MatrixBenchmark;
import com.jalllm.geforce.config.ModelConfig;
import com.jalllm.geforce.config.RuntimeConfig;
import com.jalllm.geforce.config.TrainingConfig;
import com.jalllm.geforce.generation.GreedySampler;
import com.jalllm.geforce.generation.Sampler;
import com.jalllm.geforce.generation.TemperatureSampler;
import com.jalllm.geforce.generation.TopKSampler;
import com.jalllm.geforce.generation.TopPSampler;
import com.jalllm.geforce.math.MatrixMath;
import com.jalllm.geforce.model.MiniGPT;
import com.jalllm.geforce.optimizer.AdamW;
import com.jalllm.geforce.quantization.Int4Quantizer;
import com.jalllm.geforce.quantization.Int8Quantizer;
import com.jalllm.geforce.quantization.QuantizationType;
import com.jalllm.geforce.quantization.QuantizedTensor;
import com.jalllm.geforce.quantization.Quantizer;
import com.jalllm.geforce.autograd.Variable;
import com.jalllm.geforce.serialization.ModelReader;
import com.jalllm.geforce.serialization.ModelWriter;
import com.jalllm.geforce.tensor.Tensor;
import com.jalllm.geforce.tokenizer.BPETokenizer;
import com.jalllm.geforce.tokenizer.CharacterTokenizer;
import com.jalllm.geforce.tokenizer.Tokenizer;
import com.jalllm.geforce.tokenizer.WordTokenizer;
import com.jalllm.geforce.training.Batch;
import com.jalllm.geforce.training.Trainer;
import com.jalllm.geforce.training.TrainingState;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/** Command surface for the JALLM CLI. */
final class CliDispatcher {
    private CliDispatcher() { }

    static void dispatch(String[] args) {
        if (args.length == 0) {
            Main.printUsage();
            return;
        }

        switch (args[0]) {
            case "help", "--help", "-h" -> Main.printUsage();
            case "selftest" -> runSelfTest();
            case "train" -> runTrain(args);
            case "generate" -> runGenerate(args);
            case "evaluate" -> runEvaluate(args);
            case "inspect" -> runInspect(args);
            case "benchmark" -> runBenchmark(args);
            case "quantize" -> runQuantize(args);
            default -> {
                System.err.println("Unknown command: " + args[0]);
                Main.printUsage();
            }
        }
    }

    private static void runSelfTest() {
        Tensor values = new Tensor(new int[] {2, 3}, new float[] {1, 2, 3, 4, 5, 6});
        Tensor reshaped = values.reshape(3, 2);
        require(reshaped.get(2, 1) == 6.0f, "reshape preserves row-major data");

        Tensor left = new Tensor(new int[] {2, 2}, new float[] {1, 2, 3, 4});
        Tensor right = new Tensor(new int[] {2, 2}, new float[] {5, 6, 7, 8});
        Tensor product = MatrixMath.matmul(left, right);
        require(product.get(0, 0) == 19.0f && product.get(1, 1) == 50.0f, "matrix multiplication");

        Tensor probabilities = Softmax.forward(new Tensor(new int[] {3}, new float[] {1000, 1001, 1002}));
        float sum = probabilities.get(0) + probabilities.get(1) + probabilities.get(2);
        require(Math.abs(sum - 1.0f) < 1e-5f, "numerically stable softmax");
        System.out.println("Tensor engine self-test: PASSED");
    }

    private static void runTrain(String[] args) {
        if (args.length < 2) {
            printTrainUsage();
            return;
        }

        String dataPath = args[1];
        String outputDir = args.length > 2 ? args[2] : "models/checkpoints";

        try {
            // Load training data
            String text = Files.readString(Paths.get(dataPath));
            System.out.println("Loaded training data: " + text.length() + " characters");

            // Create tokenizer
            Tokenizer tokenizer = new CharacterTokenizer();
            int[] tokens = toIntArray(tokenizer.encode(text));
            System.out.println("Tokenized: " + tokens.length + " tokens, vocab size: " + tokenizer.vocabSize());

            // Model config
            ModelConfig modelConfig = new ModelConfig(
                    tokenizer.vocabSize(), 256, 128, 4, 4, 512, 64, 0.1f
            );

            // Training config
            TrainingConfig trainConfig = new TrainingConfig(
                    1000, 4, 1e-3f, 0.01f, 1.0f, 100, outputDir
            );

            // Runtime config
            RuntimeConfig runtimeConfig = new RuntimeConfig(4, false, 42);

            // Create model and trainer
            MiniGPT model = new MiniGPT(modelConfig);
            AdamW optimizer = new AdamW(model.parameters(), trainConfig.learningRate(), trainConfig.weightDecay());
            Trainer trainer = new Trainer(model, optimizer, trainConfig, runtimeConfig);

            // Create batches
            List<Batch> batches = createBatches(tokens, trainConfig.batchSize(), modelConfig.maxSeqLen(), modelConfig.numHeads());
            System.out.println("Created " + batches.size() + " training batches");

            // Train
            TrainingState state = trainer.train(batches, trainConfig.maxSteps());
            System.out.println("Training completed. Final loss: " + state.loss());

            // Save model
            File outDir = new File(outputDir);
            outDir.mkdirs();
            ModelWriter.write(model, new File(outDir, "model.bin"));
            System.out.println("Model saved to " + outputDir + "/model.bin");

        } catch (IOException e) {
            System.err.println("Error: " + e.getMessage());
        }
    }

    private static void runGenerate(String[] args) {
        if (args.length < 3) {
            printGenerateUsage();
            return;
        }

        String modelPath = args[1];
        String prompt = args[2];
        int maxTokens = args.length > 3 ? Integer.parseInt(args[3]) : 100;
        float temperature = args.length > 4 ? Float.parseFloat(args[4]) : 1.0f;
        int topK = args.length > 5 ? Integer.parseInt(args[5]) : 0;
        float topP = args.length > 6 ? Float.parseFloat(args[6]) : 1.0f;
        String samplerType = args.length > 7 ? args[7] : "temperature";

        try {
            // Load model
            MiniGPT model = ModelReader.read(new File(modelPath));
            ModelConfig config = model.config();
            System.out.println("Loaded model: " + config.embedDim() + "d, " +
                    config.numLayers() + "L, " + config.numHeads() + "H");

            // Create tokenizer (must match training)
            Tokenizer tokenizer = new CharacterTokenizer();

            // Encode prompt
            int[] promptTokens = toIntArray(tokenizer.encode(prompt));
            System.out.println("Prompt tokens: " + promptTokens.length);

            // Create sampler
            // MiniGPT currently owns the sampling loop; validate the requested mode here.
            createSampler(samplerType, temperature, topK, topP);

            // Generate
            int[] generated = model.generate(promptTokens, maxTokens, temperature, topK, topP);
            String output = tokenizer.decode(java.util.Arrays.stream(generated).boxed().toList());
            System.out.println("\n--- Generated ---");
            System.out.println(output);

        } catch (IOException e) {
            System.err.println("Error: " + e.getMessage());
        }
    }

    private static void runEvaluate(String[] args) {
        if (args.length < 3) {
            printEvaluateUsage();
            return;
        }

        String modelPath = args[1];
        String dataPath = args[2];
        int batchSize = args.length > 3 ? Integer.parseInt(args[3]) : 4;

        try {
            // Load model
            MiniGPT model = ModelReader.read(new File(modelPath));
            ModelConfig config = model.config();
            System.out.println("Loaded model: " + config.embedDim() + "d, " +
                    config.numLayers() + "L, " + config.numHeads() + "H");

            // Load evaluation data
            String text = Files.readString(Paths.get(dataPath));
            Tokenizer tokenizer = new CharacterTokenizer();
            int[] tokens = toIntArray(tokenizer.encode(text));

            // Create batches
            List<Batch> batches = createBatches(tokens, batchSize, config.maxSeqLen(), config.numHeads());
            System.out.println("Created " + batches.size() + " evaluation batches");

            // Evaluate
            double totalLoss = 0;
            int count = 0;
            for (Batch batch : batches) {
                Tensor logits = model.forward(batch.inputIds(), batch.mask());
                totalLoss += new com.jalllm.geforce.loss.CrossEntropyLoss().forward(logits, batch.targetIds()).data()[0];
                count++;
            }
            double meanLoss = count == 0 ? Double.NaN : totalLoss / count;
            System.out.printf("Evaluation completed on %d batches. Loss: %.6f, perplexity: %.4f%n", count, meanLoss, Math.exp(Math.min(meanLoss, 20.0)));

        } catch (IOException e) {
            System.err.println("Error: " + e.getMessage());
        }
    }

    private static void runInspect(String[] args) {
        if (args.length < 2) {
            printInspectUsage();
            return;
        }

        String modelPath = args[1];

        try {
            MiniGPT model = ModelReader.read(new File(modelPath));
            ModelConfig config = model.config();

            System.out.println("=== Model Inspection ===");
            System.out.println("Architecture: MiniGPT (Decoder-only Transformer)");
            System.out.println("Embedding dim: " + config.embedDim());
            System.out.println("Layers: " + config.numLayers());
            System.out.println("Heads: " + config.numHeads());
            System.out.println("Head dim: " + config.headDim());
            System.out.println("FFN dim: " + config.ffDim());
            System.out.println("Vocab size: " + config.vocabSize());
            System.out.println("Max seq len: " + config.maxSeqLen());
            System.out.println("Dropout: " + config.dropout());

            // Count parameters
            long totalParams = model.parameters().stream()
                    .mapToLong(p -> p.value().numel())
                    .sum();
            System.out.println("Total parameters: " + totalParams + " (" + String.format("%.2f", totalParams / 1e6) + "M)");

            // Memory estimate
            double fp32MB = totalParams * 4 / 1e6;
            double int8MB = totalParams * 1 / 1e6;
            double int4MB = totalParams * 0.5 / 1e6;
            System.out.println("Memory (FP32): " + String.format("%.1f", fp32MB) + " MB");
            System.out.println("Memory (INT8): " + String.format("%.1f", int8MB) + " MB");
            System.out.println("Memory (INT4): " + String.format("%.1f", int4MB) + " MB");

        } catch (IOException e) {
            System.err.println("Error: " + e.getMessage());
        }
    }

    private static void runBenchmark(String[] args) {
        if (args.length < 2) {
            printBenchmarkUsage();
            return;
        }

        String type = args[1];

        switch (type) {
            case "matrix" -> MatrixBenchmark.runSuite();
            case "inference" -> InferenceBenchmark.runSuite();
            case "all" -> {
                MatrixBenchmark.runSuite();
                System.out.println();
                InferenceBenchmark.runSuite();
            }
            default -> {
                System.err.println("Unknown benchmark type: " + type);
                printBenchmarkUsage();
            }
        }
    }

    private static void runQuantize(String[] args) {
        if (args.length < 4) {
            printQuantizeUsage();
            return;
        }

        String modelPath = args[1];
        String outputPath = args[2];
        String quantType = args[3];

        try {
            MiniGPT model = ModelReader.read(new File(modelPath));
            System.out.println("Loaded model: " + model.parameters().size() + " parameter tensors");

            QuantizationType type = switch (quantType.toLowerCase()) {
                case "int8" -> QuantizationType.INT8;
                case "int4" -> QuantizationType.INT4;
                default -> throw new IllegalArgumentException("Unknown quantization type: " + quantType);
            };

            // Quantize all parameters
            Quantizer quantizer = (type == QuantizationType.INT8) ? new Int8Quantizer() : new Int4Quantizer();
            for (Variable param : model.parameters()) {
                QuantizedTensor qt = quantizer.quantize(param.value());
                // Note: In a real implementation, we'd replace the parameter with the quantized version
                // For now, we just demonstrate the quantization
                Tensor dequantized = quantizer.dequantize(qt);
                System.arraycopy(dequantized.data(), 0, param.value().data(), 0, param.value().data().length);
            }

            // Save quantized model
            ModelWriter.write(model, new File(outputPath));
            System.out.println("Quantized model saved to " + outputPath);

        } catch (IOException e) {
            System.err.println("Error: " + e.getMessage());
        }
    }

    private static List<Batch> createBatches(int[] tokens, int batchSize, int seqLen, int numHeads) {
        List<Batch> batches = new ArrayList<>();
        int numTokens = tokens.length - seqLen;
        int numBatches = numTokens / batchSize;

        for (int b = 0; b < numBatches; b++) {
            float[][] inputData = new float[batchSize][seqLen];
            float[][] targetData = new float[batchSize][seqLen];

            for (int i = 0; i < batchSize; i++) {
                int start = b * batchSize + i;
                for (int s = 0; s < seqLen; s++) {
                    inputData[i][s] = tokens[start + s];
                    targetData[i][s] = tokens[start + s + 1];
                }
            }

            Tensor inputs = new Tensor(new int[]{batchSize, seqLen}, flatten(inputData));
            Tensor targets = new Tensor(new int[]{batchSize, seqLen}, flatten(targetData));
            batches.add(new Batch(inputs, targets, com.jalllm.geforce.attention.CausalMask.create(batchSize, numHeads, seqLen)));
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

    private static int[] toIntArray(List<Integer> values) {
        return values.stream().mapToInt(Integer::intValue).toArray();
    }

    private static Sampler createSampler(String type, float temperature, int topK, float topP) {
        return switch (type.toLowerCase()) {
            case "greedy" -> new GreedySampler();
            case "temperature" -> new TemperatureSampler(temperature);
            case "topk" -> new TopKSampler(topK, temperature);
            case "topp" -> new TopPSampler(topP, temperature);
            default -> new TemperatureSampler(temperature);
        };
    }

    private static void require(boolean condition, String check) {
        if (!condition) {
            throw new IllegalStateException("Self-test failed: " + check);
        }
    }

    private static void printTrainUsage() {
        System.out.println("Usage: jallm train <data-file> [output-dir]");
        System.out.println("  data-file: Path to training text file");
        System.out.println("  output-dir: Directory to save checkpoints (default: models/checkpoints)");
    }

    private static void printGenerateUsage() {
        System.out.println("Usage: jallm generate <model-file> <prompt> [max-tokens] [temperature] [top-k] [top-p] [sampler]");
        System.out.println("  model-file: Path to trained model (.bin)");
        System.out.println("  prompt: Text prompt to start generation");
        System.out.println("  max-tokens: Maximum tokens to generate (default: 100)");
        System.out.println("  temperature: Sampling temperature (default: 1.0)");
        System.out.println("  top-k: Top-k sampling (default: 0 = disabled)");
        System.out.println("  top-p: Top-p sampling (default: 1.0 = disabled)");
        System.out.println("  sampler: greedy|temperature|topk|topp (default: temperature)");
    }

    private static void printEvaluateUsage() {
        System.out.println("Usage: jallm evaluate <model-file> <data-file> [batch-size]");
    }

    private static void printInspectUsage() {
        System.out.println("Usage: jallm inspect <model-file>");
    }

    private static void printBenchmarkUsage() {
        System.out.println("Usage: jallm benchmark <matrix|inference|all>");
    }

    private static void printQuantizeUsage() {
        System.out.println("Usage: jallm quantize <model-file> <output-file> <int8|int4>");
    }
}
