package com.jalllm.geforce.serialization;

import com.jalllm.geforce.autograd.Variable;
import com.jalllm.geforce.model.LanguageModel;
import com.jalllm.geforce.model.MiniGPT;
import com.jalllm.geforce.tensor.Tensor;
import com.jalllm.geforce.training.TrainingState;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;
import java.util.zip.GZIPInputStream;

/**
 * Reads model checkpoints from disk.
 */
public final class ModelReader {

    private ModelReader() {}

    /**
     * Loads a model checkpoint.
     */
    public static Checkpoint loadCheckpoint(String path) throws IOException {
        try (DataInputStream in = new DataInputStream(
                new BufferedInputStream(new GZIPInputStream(new FileInputStream(path))))) {

            // Verify magic number
            int magic = in.readInt();
            if (magic != 0x4A4C4C4D) {
                throw new IOException("Invalid checkpoint file: wrong magic number");
            }

            // Version
            int version = in.readInt();
            if (version != 1 && version != 2) {
                throw new IOException("Unsupported checkpoint version: " + version);
            }

            // Model config
            String modelType = in.readUTF();
            int vocabSize = in.readInt();
            int embedDim = in.readInt();
            int maxSeqLen = in.readInt();
            int numLayers = in.readInt();
            int numHeads = in.readInt();
            int ffDim = version >= 2 ? in.readInt() : embedDim * 4;
            float dropout = version >= 2 ? in.readFloat() : 0.1f;

            // Training state
            TrainingState state = readTrainingState(in);

            // Create model
            LanguageModel model;
            if ("MiniGPT".equals(modelType)) {
                model = new MiniGPT(new com.jalllm.geforce.config.ModelConfig(embedDim, maxSeqLen, numLayers, numHeads, ffDim, vocabSize, embedDim / numHeads, dropout));
            } else {
                throw new IOException("Unknown model type: " + modelType);
            }

            // Load parameters
            int numParams = in.readInt();
            List<Variable> params = model.parameters();
            if (params.size() != numParams) {
                throw new IOException("Parameter count mismatch: expected " + params.size() + ", got " + numParams);
            }

            for (int i = 0; i < numParams; i++) {
                Variable param = readVariable(in);
                // Copy data into model parameter
                System.arraycopy(param.value().data(), 0, params.get(i).value().data(), 0, param.value().data().length);
            }

            return new Checkpoint(model, state);
        }
    }

    /**
     * Loads only model parameters (for inference).
     */
    public static LanguageModel loadModel(String path) throws IOException {
        try (DataInputStream in = new DataInputStream(
                new BufferedInputStream(new GZIPInputStream(new FileInputStream(path))))) {

            int magic = in.readInt();
            if (magic != 0x4A4C4C4D) {
                throw new IOException("Invalid model file: wrong magic number");
            }

            int version = in.readInt();
            if (version != 1 && version != 2) {
                throw new IOException("Unsupported model version: " + version);
            }

            String modelType = in.readUTF();
            int vocabSize = in.readInt();
            int embedDim = in.readInt();
            int maxSeqLen = in.readInt();
            int numLayers = in.readInt();
            int numHeads = in.readInt();
            int ffDim = version >= 2 ? in.readInt() : embedDim * 4;
            float dropout = version >= 2 ? in.readFloat() : 0.1f;

            LanguageModel model;
            if ("MiniGPT".equals(modelType)) {
                model = new MiniGPT(new com.jalllm.geforce.config.ModelConfig(embedDim, maxSeqLen, numLayers, numHeads, ffDim, vocabSize, embedDim / numHeads, dropout));
            } else {
                throw new IOException("Unknown model type: " + modelType);
            }

            int numParams = in.readInt();
            List<Variable> params = model.parameters();
            if (params.size() != numParams) {
                throw new IOException("Parameter count mismatch: expected " + params.size() + ", got " + numParams);
            }

            for (int i = 0; i < numParams; i++) {
                Variable param = readVariable(in);
                System.arraycopy(param.value().data(), 0, params.get(i).value().data(), 0, param.value().data().length);
            }

            return model;
        }
    }

    private static TrainingState readTrainingState(DataInputStream in) throws IOException {
        TrainingState state = new TrainingState();
        state.setEpoch(in.readInt());
        state.setStep(in.readInt());
        state.setGlobalStep(in.readInt());
        state.setLearningRate(in.readFloat());
        state.setTrainLoss(in.readFloat());
        state.setValLoss(in.readFloat());
        state.setPerplexity(in.readFloat());
        state.setTimestamp(java.time.Instant.ofEpochMilli(in.readLong()));
        state.setModelConfigJson(in.readUTF());
        return state;
    }

    private static Variable readVariable(DataInputStream in) throws IOException {
        String name = in.readUTF();

        int rank = in.readInt();
        int[] shape = new int[rank];
        for (int i = 0; i < rank; i++) {
            shape[i] = in.readInt();
        }

        int length = in.readInt();
        byte[] bytes = new byte[length * 4];
        in.readFully(bytes);

        ByteBuffer buffer = ByteBuffer.wrap(bytes);
        buffer.order(ByteOrder.LITTLE_ENDIAN);
        float[] data = new float[length];
        for (int i = 0; i < length; i++) {
            data[i] = buffer.getFloat();
        }

        Tensor tensor = new Tensor(shape, data);
        return Variable.of(tensor, name);
    }

    /**
     * Container for loaded checkpoint.
     */
    public static final class Checkpoint {
        private final LanguageModel model;
        private final TrainingState state;

        public Checkpoint(LanguageModel model, TrainingState state) {
            this.model = model;
            this.state = state;
        }

        public LanguageModel model() { return model; }
        public TrainingState state() { return state; }
    }

    /**
     * Reads a MiniGPT model from a file.
     */
    public static com.jalllm.geforce.model.MiniGPT read(java.io.File file) throws IOException {
        return (com.jalllm.geforce.model.MiniGPT) loadModel(file.getAbsolutePath());
    }
}
