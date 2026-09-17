package com.jalllm.geforce.math;

import com.jalllm.geforce.tensor.Tensor;
import com.jalllm.geforce.tensor.Tensor2D;
import com.jalllm.geforce.tensor.TensorShape;

import java.util.stream.IntStream;

/**
 * High-performance matrix mathematics engine.
 * Includes cache-conscious, parallelized matrix multiplication.
 */
public final class MatrixMath {

    private MatrixMath() {}

    /**
     * Performs matrix multiplication: C = A * B.
     * Supports 2D matrices and higher-dimensional batched tensors.
     */
    public static Tensor matmul(Tensor A, Tensor B) {
        int rA = A.rank();
        int rB = B.rank();

        if (rA == 2 && rB == 2) {
            return matmul2D(A, B);
        } else if (rA >= 2 && rB >= 2) {
            return matmulBatched(A, B);
        } else {
            throw new IllegalArgumentException("Unsupported shapes for matmul: " + 
                    TensorShape.toString(A.shape()) + " and " + TensorShape.toString(B.shape()));
        }
    }

    /**
     * Cache-conscious parallelized 2D matrix multiplication.
     * Uses IKJ loop ordering for sequential memory access.
     */
    public static Tensor matmul2D(Tensor A, Tensor B) {
        if (A.rank() != 2 || B.rank() != 2) {
            throw new IllegalArgumentException("matmul2D requires rank-2 tensors: "
                    + TensorShape.toString(A.shape()) + " and " + TensorShape.toString(B.shape()));
        }
        int[] shapeA = A.shape();
        int[] shapeB = B.shape();

        int M = shapeA[0];
        int K = shapeA[1];
        int N = shapeB[1];

        if (K != shapeB[0]) {
            throw new IllegalArgumentException("Incompatible shapes for matmul2D: " + 
                    TensorShape.toString(shapeA) + " and " + TensorShape.toString(shapeB));
        }

        float[] dataA = A.data();
        float[] dataB = B.data();
        float[] dataC = new float[M * N];

        // Parallelize over rows of A if size is significant
        boolean parallel = (M * N * K) > 65536; // threshold for parallel execution

        if (parallel) {
            IntStream.range(0, M).parallel().forEach(i -> {
                int rowOffsetA = i * K;
                int rowOffsetC = i * N;
                for (int k = 0; k < K; k++) {
                    float valA = dataA[rowOffsetA + k];
                    if (valA == 0.0f) continue;
                    int rowOffsetB = k * N;
                    for (int j = 0; j < N; j++) {
                        dataC[rowOffsetC + j] += valA * dataB[rowOffsetB + j];
                    }
                }
            });
        } else {
            for (int i = 0; i < M; i++) {
                int rowOffsetA = i * K;
                int rowOffsetC = i * N;
                for (int k = 0; k < K; k++) {
                    float valA = dataA[rowOffsetA + k];
                    if (valA == 0.0f) continue;
                    int rowOffsetB = k * N;
                    for (int j = 0; j < N; j++) {
                        dataC[rowOffsetC + j] += valA * dataB[rowOffsetB + j];
                    }
                }
            }
        }

        return new Tensor(new int[]{M, N}, dataC);
    }

    /**
     * Performs batched matrix multiplication.
     * Processes batch dimensions sequentially and 2D slices using matmul2D.
     */
    public static Tensor matmulBatched(Tensor A, Tensor B) {
        int[] shapeA = A.shape();
        int[] shapeB = B.shape();

        int rankA = A.rank();
        int rankB = B.rank();

        int batchRank = Math.max(rankA, rankB) - 2;
        int[] batchShape = new int[batchRank];
        for (int i = 0; i < batchRank; i++) {
            int aAxis = i - (batchRank - (rankA - 2));
            int bAxis = i - (batchRank - (rankB - 2));
            int aDim = aAxis < 0 ? 1 : shapeA[aAxis];
            int bDim = bAxis < 0 ? 1 : shapeB[bAxis];
            if (aDim != bDim && aDim != 1 && bDim != 1) throw new IllegalArgumentException("Incompatible batch dimensions: " + TensorShape.toString(shapeA) + " and " + TensorShape.toString(shapeB));
            batchShape[i] = Math.max(aDim, bDim);
        }

        int M = shapeA[rankA - 2];
        int K = shapeA[rankA - 1];
        int N = shapeB[rankB - 1];

        if (K != shapeB[rankB - 2]) {
            throw new IllegalArgumentException("Inner dimensions must match: " + K + " != " + shapeB[rankB - 2]);
        }

        int numBatches = (int) TensorShape.size(batchShape);

        int sliceSizeA = M * K;
        int sliceSizeB = K * N;
        int sliceSizeC = M * N;

        float[] dataA = A.data();
        float[] dataB = B.data();
        float[] dataC = new float[numBatches * sliceSizeC];

        int[] outShape = java.util.Arrays.copyOf(batchShape, batchRank + 2);
        outShape[batchRank] = M;
        outShape[batchRank + 1] = N;

        // Run batch slices
        for (int b = 0; b < numBatches; b++) {
            final int batchIdx = b;
            int offsetA = broadcastBatchOffset(batchIdx, batchShape, shapeA, sliceSizeA);
            int offsetB = broadcastBatchOffset(batchIdx, batchShape, shapeB, sliceSizeB);
            int offsetC = batchIdx * sliceSizeC;

            // We can reuse or directly run 2D matmul on slices
            for (int i = 0; i < M; i++) {
                int rA = offsetA + i * K;
                int rC = offsetC + i * N;
                for (int k = 0; k < K; k++) {
                    float valA = dataA[rA + k];
                    if (valA == 0.0f) continue;
                    int rB = offsetB + k * N;
                    for (int j = 0; j < N; j++) {
                        dataC[rC + j] += valA * dataB[rB + j];
                    }
                }
            }
        }

        return new Tensor(outShape, dataC);
    }

    /**
     * Transposes a 2D matrix.
     */
    public static Tensor transpose(Tensor A) {
        if (A.rank() < 2) throw new IllegalArgumentException("Transpose requires rank >= 2");
        int[] shape = A.shape();
        int rows = shape[shape.length - 2], cols = shape[shape.length - 1];
        int batches = A.elementCount() / (rows * cols);
        float[] res = new float[A.elementCount()], data = A.data();
        for (int b = 0; b < batches; b++) for (int r = 0; r < rows; r++) for (int c = 0; c < cols; c++) res[(b * cols + c) * rows + r] = data[(b * rows + r) * cols + c];
        shape[shape.length - 2] = cols;
        shape[shape.length - 1] = rows;
        return new Tensor(shape, res);
    }

    private static int broadcastBatchOffset(int flatBatch, int[] outputBatchShape, int[] tensorShape, int sliceSize) {
        int tensorBatchRank = tensorShape.length - 2;
        int offset = 0, stride = 1, remaining = flatBatch;
        for (int outAxis = outputBatchShape.length - 1; outAxis >= 0; outAxis--) {
            int coord = remaining % outputBatchShape[outAxis];
            remaining /= outputBatchShape[outAxis];
            int tensorAxis = outAxis - (outputBatchShape.length - tensorBatchRank);
            if (tensorAxis >= 0) {
                int dim = tensorShape[tensorAxis];
                offset += (dim == 1 ? 0 : coord) * stride;
                stride *= dim;
            }
        }
        return offset * sliceSize;
    }
}
