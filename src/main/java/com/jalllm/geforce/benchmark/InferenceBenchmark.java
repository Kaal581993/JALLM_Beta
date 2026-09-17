package com.jalllm.geforce.benchmark;

import com.jalllm.geforce.config.ModelConfig;
import com.jalllm.geforce.config.RuntimeConfig;
import com.jalllm.geforce.model.MiniGPT;
import com.jalllm.geforce.tensor.Tensor;
import com.jalllm.geforce.tokenizer.CharacterTokenizer;
import com.jalllm.geforce.tokenizer.Tokenizer;

import java.util.Random;

/**
 * Inference benchmarks for the MiniGPT model.
 */
public final class InferenceBenchmark {

    private InferenceBenchmark() {}

    /**
     * Benchmarks forward pass of MiniGPT.
     */
    public static BenchmarkResult benchmarkForward(ModelConfig config, int batchSize, int seqLen, int iterations, int warmup) {
        MiniGPT model = new MiniGPT(config);
        Tensor input = randomIntTensor(batchSize, seqLen, config.vocabSize());

        // Warmup
        for (int i = 0; i < warmup; i++) {
            model.forward(input, model.createCausalMask(batchSize, seqLen));
        }

        // Benchmark
        long start = System.nanoTime();
        for (int i = 0; i < iterations; i++) {
            model.forward(input, model.createCausalMask(batchSize, seqLen));
        }
        long end = System.nanoTime();

        double elapsedMs = (end - start) / 1_000_000.0;
        double avgMs = elapsedMs / iterations;
        double tokensPerSec = (batchSize * seqLen) / (avgMs / 1000.0);

        return new BenchmarkResult("forward", config, batchSize, seqLen, iterations, avgMs, tokensPerSec);
    }

    /**
     * Benchmarks autoregressive generation.
     */
    public static BenchmarkResult benchmarkGenerate(ModelConfig config, int promptLen, int genLen, int iterations, int warmup) {
        MiniGPT model = new MiniGPT(config);
        Tokenizer tokenizer = new CharacterTokenizer();
        String prompt = "The quick brown fox jumps over the lazy dog. ".repeat(promptLen / 44 + 1);
        int[] promptTokens = tokenizer.encode(prompt.substring(0, Math.min(prompt.length(), promptLen))).stream().mapToInt(Integer::intValue).toArray();

        // Warmup
        for (int i = 0; i < warmup; i++) {
            model.generate(promptTokens, genLen, 1.0f, 0, 1.0f);
        }

        // Benchmark
        long start = System.nanoTime();
        for (int i = 0; i < iterations; i++) {
            model.generate(promptTokens, genLen, 1.0f, 0, 1.0f);
        }
        long end = System.nanoTime();

        double elapsedMs = (end - start) / 1_000_000.0;
        double avgMs = elapsedMs / iterations;
        double tokensPerSec = genLen / (avgMs / 1000.0);

        return new BenchmarkResult("generate", config, 1, promptLen + genLen, iterations, avgMs, tokensPerSec);
    }

    /**
     * Runs a suite of inference benchmarks.
     */
    public static void runSuite() {
        System.out.println("=== Inference Benchmarks ===\n");

        // Small model
        ModelConfig small = new ModelConfig(1000, 128, 128, 4, 4);
        runAndPrint(() -> benchmarkForward(small, 1, 128, 50, 10));
        runAndPrint(() -> benchmarkForward(small, 4, 128, 20, 5));
        runAndPrint(() -> benchmarkGenerate(small, 32, 64, 10, 3));

        // Medium model
        ModelConfig medium = new ModelConfig(8000, 256, 256, 6, 8);
        runAndPrint(() -> benchmarkForward(medium, 1, 256, 20, 5));
        runAndPrint(() -> benchmarkForward(medium, 2, 256, 10, 3));
        runAndPrint(() -> benchmarkGenerate(medium, 64, 128, 5, 2));

        // Large model (if memory permits)
        ModelConfig large = new ModelConfig(16000, 512, 512, 8, 12);
        runAndPrint(() -> benchmarkForward(large, 1, 512, 10, 3));
        runAndPrint(() -> benchmarkGenerate(large, 128, 256, 3, 1));
    }

    private static void runAndPrint(java.util.function.Supplier<BenchmarkResult> bench) {
        try {
            BenchmarkResult result = bench.get();
            System.out.printf("%s: %.2f ms/iter, %.0f tokens/sec\n",
                    result.name(), result.avgMs(), result.tokensPerSec());
        } catch (OutOfMemoryError e) {
            System.out.println("OOM - skipping larger benchmark");
        }
    }

    private static Tensor randomIntTensor(int batch, int seqLen, int vocabSize) {
        float[] data = new float[batch * seqLen];
        Random r = new Random(42);
        for (int i = 0; i < data.length; i++) {
            data[i] = r.nextInt(vocabSize);
        }
        return new Tensor(new int[]{batch, seqLen}, data);
    }

    /**
     * Benchmark result container.
     */
    public static final class BenchmarkResult {
        private final String name;
        private final ModelConfig config;
        private final int batchSize;
        private final int seqLen;
        private final int iterations;
        private final double avgMs;
        private final double tokensPerSec;

        public BenchmarkResult(String name, ModelConfig config, int batchSize, int seqLen,
                               int iterations, double avgMs, double tokensPerSec) {
            this.name = String.format("%s(model_d=%d_L=%d_H=%d, batch=%d, seq=%d)",
                    name, config.embedDim(), config.numLayers(), config.numHeads(), batchSize, seqLen);
            this.config = config;
            this.batchSize = batchSize;
            this.seqLen = seqLen;
            this.iterations = iterations;
            this.avgMs = avgMs;
            this.tokensPerSec = tokensPerSec;
        }

        public String name() { return name; }
        public ModelConfig config() { return config; }
        public int batchSize() { return batchSize; }
        public int seqLen() { return seqLen; }
        public int iterations() { return iterations; }
        public double avgMs() { return avgMs; }
        public double tokensPerSec() { return tokensPerSec; }
    }
}
