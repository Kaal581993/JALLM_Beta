package com.jalllm.geforce.serialization;

import com.jalllm.geforce.autograd.Variable;
import com.jalllm.geforce.model.LanguageModel;
import com.jalllm.geforce.tensor.Tensor;
import com.jalllm.geforce.training.TrainingState;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.List;
import java.util.zip.GZIPOutputStream;

/**
 * Writes model checkpoints to disk.
 * Format: Custom binary format with gzip compression.
 */
public final class ModelWriter {

    private ModelWriter() {}

    /**
     * Saves a model checkpoint.
     */
    public static void saveCheckpoint(LanguageModel model, TrainingState state, String path) throws IOException {
        try (DataOutputStream out = new DataOutputStream(
                new BufferedOutputStream(new GZIPOutputStream(new FileOutputStream(path))))) {

            // Magic number
            out.writeInt(0x4A4C4C4D); // "JLLM"

            // Version
            out.writeInt(2);

            // Model config
            writeModelConfig(model, out);

            // Training state
            writeTrainingState(state, out);

            // Parameters
            List<Variable> params = model.parameters();
            out.writeInt(params.size());

            for (Variable param : params) {
                writeVariable(param, out);
            }
        }
    }

    /**
     * Saves only model parameters (for inference).
     */
    public static void saveModel(LanguageModel model, String path) throws IOException {
        try (DataOutputStream out = new DataOutputStream(
                new BufferedOutputStream(new GZIPOutputStream(new FileOutputStream(path))))) {

            out.writeInt(0x4A4C4C4D); // "JLLM"
            out.writeInt(2);

            writeModelConfig(model, out);

            List<Variable> params = model.parameters();
            out.writeInt(params.size());

            for (Variable param : params) {
                writeVariable(param, out);
            }
        }
    }

    private static void writeModelConfig(LanguageModel model, DataOutputStream out) throws IOException {
        if (model instanceof com.jalllm.geforce.model.MiniGPT) {
            com.jalllm.geforce.model.MiniGPT gpt = (com.jalllm.geforce.model.MiniGPT) model;
            out.writeUTF("MiniGPT");
            out.writeInt(gpt.vocabSize());
            out.writeInt(gpt.embedDim());
            out.writeInt(gpt.maxSeqLen());
            out.writeInt(gpt.numLayers());
            out.writeInt(gpt.numHeads());
            out.writeInt(gpt.ffDim());
            out.writeFloat(gpt.dropout());
        } else {
            out.writeUTF("Unknown");
            out.writeInt(model.vocabSize());
            out.writeInt(model.embedDim());
            out.writeInt(model.maxSeqLen());
            out.writeInt(0);
            out.writeInt(0);
            out.writeInt(model.embedDim() * 4);
            out.writeFloat(0.1f);
        }
    }

    private static void writeTrainingState(TrainingState state, DataOutputStream out) throws IOException {
        out.writeInt(state.epoch());
        out.writeInt(state.step());
        out.writeInt(state.globalStep());
        out.writeFloat(state.learningRate());
        out.writeFloat(state.trainLoss());
        out.writeFloat(state.valLoss());
        out.writeFloat(state.perplexity());
        out.writeLong(state.timestamp().toEpochMilli());
        out.writeUTF(state.modelConfigJson() != null ? state.modelConfigJson() : "");
    }

    private static void writeVariable(Variable var, DataOutputStream out) throws IOException {
        Tensor tensor = var.value();
        out.writeUTF(var.name());

        int[] shape = tensor.shape();
        out.writeInt(shape.length);
        for (int dim : shape) {
            out.writeInt(dim);
        }

        float[] data = tensor.data();
        out.writeInt(data.length);

        // Write as bytes for efficiency
        ByteBuffer buffer = ByteBuffer.allocate(data.length * 4);
        buffer.order(ByteOrder.LITTLE_ENDIAN);
        for (float f : data) {
            buffer.putFloat(f);
        }
        out.write(buffer.array());
    }

    /**
     * Writes a MiniGPT model to a file.
     */
    public static void write(com.jalllm.geforce.model.MiniGPT model, java.io.File file) throws IOException {
        saveModel(model, file.getAbsolutePath());
    }
}
