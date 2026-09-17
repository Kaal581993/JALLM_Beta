package com.jalllm.geforce.config;

import java.io.Serializable;

/**
 * Runtime configuration for inference and benchmarking.
 */
public final class RuntimeConfig implements Serializable {

    private static final long serialVersionUID = 1L;

    private int numThreads = Runtime.getRuntime().availableProcessors();
    private boolean useParallel = true;
    private int parallelThreshold = 65536; // Elements threshold for parallel matmul
    private String device = "cpu"; // cpu, cuda (future)
    private int deviceId = 0;
    private boolean enableQuantization = false;
    private String quantizationType = "int8"; // int8, int4
    private int maxBatchSize = 32;
    private int maxSeqLen = 2048;
    private boolean kvCache = true; // Not implemented yet
    private String dtype = "fp32"; // fp32, fp16, bf16

    public RuntimeConfig() {}

    public RuntimeConfig(int numThreads, boolean useParallel, int seed) {
        this.numThreads = numThreads;
        this.useParallel = useParallel;
        // seed is not stored in RuntimeConfig
    }

    // Getters and setters
    public int numThreads() { return numThreads; }
    public void setNumThreads(int numThreads) { this.numThreads = numThreads; }

    public boolean useParallel() { return useParallel; }
    public void setUseParallel(boolean useParallel) { this.useParallel = useParallel; }

    public int parallelThreshold() { return parallelThreshold; }
    public void setParallelThreshold(int parallelThreshold) { this.parallelThreshold = parallelThreshold; }

    public String device() { return device; }
    public void setDevice(String device) { this.device = device; }

    public int deviceId() { return deviceId; }
    public void setDeviceId(int deviceId) { this.deviceId = deviceId; }

    public boolean enableQuantization() { return enableQuantization; }
    public void setEnableQuantization(boolean enableQuantization) { this.enableQuantization = enableQuantization; }

    public String quantizationType() { return quantizationType; }
    public void setQuantizationType(String quantizationType) { this.quantizationType = quantizationType; }

    public int maxBatchSize() { return maxBatchSize; }
    public void setMaxBatchSize(int maxBatchSize) { this.maxBatchSize = maxBatchSize; }

    public int maxSeqLen() { return maxSeqLen; }
    public void setMaxSeqLen(int maxSeqLen) { this.maxSeqLen = maxSeqLen; }

    public boolean kvCache() { return kvCache; }
    public void setKvCache(boolean kvCache) { this.kvCache = kvCache; }

    public String dtype() { return dtype; }
    public void setDtype(String dtype) { this.dtype = dtype; }

    @Override
    public String toString() {
        return String.format("RuntimeConfig(threads=%d, parallel=%s, device=%s, quant=%s, dtype=%s)",
                numThreads, useParallel, device, quantizationType, dtype);
    }
}