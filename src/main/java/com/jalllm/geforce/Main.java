package com.jalllm.geforce;

import com.jalllm.geforce.tensor.Tensor;
import com.jalllm.geforce.tensor.TensorShape;

import java.lang.management.ManagementFactory;
import java.lang.management.OperatingSystemMXBean;
import java.lang.management.MemoryMXBean;
import java.lang.management.MemoryUsage;

/**
 * JALLM - Java LLM From Scratch
 * <p>
 * A decoder-only Transformer language model built entirely in Java.
 * No external AI APIs, no PyTorch, no TensorFlow, no llama.cpp.
 * Everything is implemented from scratch: tensors, matrix math,
 * autograd, attention, transformer, training, tokenizer, quantization.
 */
public class Main {

    static final String NAME = "JALLM";
    static final String VERSION = "0.1";
    static final String BACKEND = "Java CPU";
    static final String PRECISION = "FP32";
    static final String EXTERNAL_AI = "NONE";

    public static void main(String[] args) {
        printBanner();
        if (args.length == 0) {
            printUsage();
            return;
        }
        CliDispatcher.dispatch(args);
    }

    static void printBanner() {
        StringBuilder sb = new StringBuilder();
        sb.append("\n");
        sb.append("=========================================\n");
        sb.append("        ").append(NAME).append(" - LLM FROM SCRATCH\n");
        sb.append("=========================================\n");
        sb.append("\n");
        sb.append("  Version:       ").append(VERSION).append("\n");
        sb.append("  Backend:       ").append(BACKEND).append("\n");
        sb.append("  Precision:     ").append(PRECISION).append("\n");
        sb.append("  External AI:   ").append(EXTERNAL_AI).append("\n");
        sb.append("  Status:        INITIALIZED\n");
        sb.append("=========================================\n");

        // Hardware info
        OperatingSystemMXBean os = ManagementFactory.getOperatingSystemMXBean();
        sb.append("\n  Hardware:\n");
        sb.append("    OS:           ").append(os.getName()).append(" ").append(os.getVersion()).append("\n");
        sb.append("    Processors:   ").append(os.getAvailableProcessors()).append(" cores\n");
        sb.append("    Arch:         ").append(os.getArch()).append("\n");

        MemoryMXBean mem = ManagementFactory.getMemoryMXBean();
        MemoryUsage heap = mem.getHeapMemoryUsage();
        long maxHeapBytes = heap.getMax();
        if (maxHeapBytes > 0) {
            sb.append("    Max Heap:     ").append(maxHeapBytes / (1024L * 1024L)).append(" MB\n");
        } else {
            sb.append("    Max Heap:     unspecified\n");
        }
        sb.append("=========================================\n");

        System.out.println(sb);

        // Quick tensor sanity check
        System.out.println("\n  Tensor engine self-check:");
        Tensor t = Tensor.zeros(2, 3);
        System.out.println("    zeros(2,3) -> shape=" + java.util.Arrays.toString(t.shape())
                + " size=" + t.size() + " elements=" + t.elementCount());
        System.out.println("    All values == 0.0: " + (t.get(0, 0) == 0.0f));
        System.out.println("=========================================\n");
    }

    static void printUsage() {
        System.out.println("\nUsage: java -jar jallm.jar <command> [options]\n");
        System.out.println("Commands:");
        System.out.println("  train      Train a new model from scratch");
        System.out.println("  generate   Generate text from a trained model");
        System.out.println("  evaluate   Evaluate a model on a dataset");
        System.out.println("  inspect    Inspect a model checkpoint");
        System.out.println("  benchmark  Run performance benchmarks");
        System.out.println("  quantize   Quantize a model (FP32 -> INT8/INT4)");
        System.out.println("  selftest   Run built-in self-tests");
        System.out.println("\nRun 'java -jar jallm.jar selftest' for a quick verification.\n");
    }
}
