package com.jalllm.geforce.benchmark;

import com.jalllm.geforce.math.MatrixMath;
import com.jalllm.geforce.tensor.Tensor;

import java.util.Random;

/**
 * Matrix multiplication benchmarks.
 */
public final class MatrixBenchmark {

    private MatrixBenchmark() {}

    /**
     * Benchmarks 2D matrix multiplication.
     */
    public static BenchmarkResult benchmarkMatmul2D(int M, int K, int N, int iterations, int warmup) {
        Tensor A = randomMatrix(M, K);
        Tensor B = randomMatrix(K, N);

        // Warmup
        for (int i = 0; i < warmup; i++) {
            MatrixMath.matmul2D(A, B);
        }

        // Benchmark
        long start = System.nanoTime();
        for (int i = 0; i < iterations; i++) {
            MatrixMath.matmul2D(A, B);
        }
        long end = System.nanoTime();

        double elapsedMs = (end - start) / 1_000_000.0;
        double avgMs = elapsedMs / iterations;
        double gflops = (2.0 * M * K * N) / (avgMs / 1000.0) / 1e9;

        return new BenchmarkResult("matmul2D", M, K, N, iterations, avgMs, gflops);
    }

    /**
     * Benchmarks batched matrix multiplication.
     */
    public static BenchmarkResult benchmarkMatmulBatched(int batch, int M, int K, int N, int iterations, int warmup) {
        Tensor A = randomMatrix(batch, M, K);
        Tensor B = randomMatrix(batch, K, N);

        for (int i = 0; i < warmup; i++) {
            MatrixMath.matmulBatched(A, B);
        }

        long start = System.nanoTime();
        for (int i = 0; i < iterations; i++) {
            MatrixMath.matmulBatched(A, B);
        }
        long end = System.nanoTime();

        double elapsedMs = (end - start) / 1_000_000.0;
        double avgMs = elapsedMs / iterations;
        double gflops = (2.0 * batch * M * K * N) / (avgMs / 1000.0) / 1e9;

        return new BenchmarkResult("matmulBatched", batch, M, K, N, iterations, avgMs, gflops);
    }

    /**
     * Runs a suite of standard benchmarks.
     */
    public static void runSuite() {
        System.out.println("=== Matrix Multiplication Benchmarks ===\n");

        // Small matrices
        runAndPrint(() -> benchmarkMatmul2D(64, 64, 64, 100, 10));
        runAndPrint(() -> benchmarkMatmul2D(128, 128, 128, 50, 10));
        runAndPrint(() -> benchmarkMatmul2D(256, 256, 256, 20, 5));
        runAndPrint(() -> benchmarkMatmul2D(512, 512, 512, 10, 3));

        // Rectangular matrices (common in transformers)
        runAndPrint(() -> benchmarkMatmul2D(128, 512, 128, 50, 10)); // QK^T
        runAndPrint(() -> benchmarkMatmul2D(128, 128, 512, 50, 10)); // Attention @ V
        runAndPrint(() -> benchmarkMatmul2D(128, 512, 2048, 20, 5)); // FFN up-projection
        runAndPrint(() -> benchmarkMatmul2D(128, 2048, 512, 20, 5)); // FFN down-projection

        // Batched
        runAndPrint(() -> benchmarkMatmulBatched(4, 128, 128, 128, 50, 10));
        runAndPrint(() -> benchmarkMatmulBatched(8, 64, 256, 64, 50, 10));
    }

    private static void runAndPrint(java.util.function.Supplier<BenchmarkResult> bench) {
        BenchmarkResult result = bench.get();
        System.out.printf("%s: %.2f ms/iter, %.2f GFLOPS\n",
                result.name(), result.avgMs(), result.gflops());
    }

    private static Tensor randomMatrix(int rows, int cols) {
        float[] data = new float[rows * cols];
        Random r = new Random(42);
        for (int i = 0; i < data.length; i++) {
            data[i] = r.nextFloat() * 2 - 1;
        }
        return new Tensor(new int[]{rows, cols}, data);
    }

    private static Tensor randomMatrix(int batch, int rows, int cols) {
        float[] data = new float[batch * rows * cols];
        Random r = new Random(42);
        for (int i = 0; i < data.length; i++) {
            data[i] = r.nextFloat() * 2 - 1;
        }
        return new Tensor(new int[]{batch, rows, cols}, data);
    }

    /**
     * Benchmark result container.
     */
    public static final class BenchmarkResult {
        private final String name;
        private final int m, k, n;
        private final int iterations;
        private final double avgMs;
        private final double gflops;

        public BenchmarkResult(String name, int m, int k, int n, int iterations, double avgMs, double gflops) {
            this.name = name + String.format("(%dx%dx%d)", m, k, n);
            this.m = m; this.k = k; this.n = n;
            this.iterations = iterations;
            this.avgMs = avgMs;
            this.gflops = gflops;
        }

        public BenchmarkResult(String name, int batch, int m, int k, int n, int iterations, double avgMs, double gflops) {
            this.name = name + String.format("(%dx%dx%dx%d)", batch, m, k, n);
            this.m = m; this.k = k; this.n = n;
            this.iterations = iterations;
            this.avgMs = avgMs;
            this.gflops = gflops;
        }

        public String name() { return name; }
        public int m() { return m; }
        public int k() { return k; }
        public int n() { return n; }
        public int iterations() { return iterations; }
        public double avgMs() { return avgMs; }
        public double gflops() { return gflops; }
    }
}