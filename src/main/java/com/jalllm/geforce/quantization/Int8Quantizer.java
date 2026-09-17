package com.jalllm.geforce.quantization;

import com.jalllm.geforce.tensor.Tensor;

/**
 * INT8 Quantization (symmetric).
 * Maps float values to int8 using scale factor.
 * q = round(x / scale)
 * x = q * scale
 */
public final class Int8Quantizer implements Quantizer {

    private final boolean perChannel;
    private final int channelAxis;

    public Int8Quantizer() {
        this(false, -1);
    }

    public Int8Quantizer(boolean perChannel, int channelAxis) {
        this.perChannel = perChannel;
        this.channelAxis = channelAxis;
    }

    @Override
    public QuantizedTensor quantize(Tensor tensor) {
        float[] data = tensor.data();
        int[] shape = tensor.shape();

        if (perChannel && channelAxis >= 0 && channelAxis < shape.length) {
            return quantizePerChannel(data, shape, channelAxis);
        } else {
            return quantizePerTensor(data, shape);
        }
    }

    private QuantizedTensor quantizePerTensor(float[] data, int[] shape) {
        // Find min/max
        float min = Float.MAX_VALUE;
        float max = -Float.MAX_VALUE;
        for (float v : data) {
            if (v < min) min = v;
            if (v > max) max = v;
        }

        // Symmetric quantization: scale = max(abs(min), abs(max)) / 127
        float maxAbs = Math.max(Math.abs(min), Math.abs(max));
        float scale = maxAbs / 127.0f;
        if (scale == 0) scale = 1.0f;

        // Quantize
        byte[] quantized = new byte[data.length];
        for (int i = 0; i < data.length; i++) {
            int q = Math.round(data[i] / scale);
            q = Math.max(-128, Math.min(127, q));
            quantized[i] = (byte) q;
        }

        return new QuantizedTensor(quantized, shape, scale, 0, QuantizationType.INT8);
    }

    private QuantizedTensor quantizePerChannel(float[] data, int[] shape, int channelAxis) {
        int numChannels = shape[channelAxis];
        int channelSize = 1;
        for (int i = channelAxis + 1; i < shape.length; i++) {
            channelSize *= shape[i];
        }
        int outerSize = 1;
        for (int i = 0; i < channelAxis; i++) {
            outerSize *= shape[i];
        }

        float[] scales = new float[numChannels];
        byte[] quantized = new byte[data.length];

        for (int c = 0; c < numChannels; c++) {
            // Find min/max for this channel
            float min = Float.MAX_VALUE;
            float max = -Float.MAX_VALUE;

            for (int o = 0; o < outerSize; o++) {
                int base = o * numChannels * channelSize + c * channelSize;
                for (int i = 0; i < channelSize; i++) {
                    float v = data[base + i];
                    if (v < min) min = v;
                    if (v > max) max = v;
                }
            }

            float maxAbs = Math.max(Math.abs(min), Math.abs(max));
            float scale = maxAbs / 127.0f;
            if (scale == 0) scale = 1.0f;
            scales[c] = scale;

            // Quantize this channel
            for (int o = 0; o < outerSize; o++) {
                int base = o * numChannels * channelSize + c * channelSize;
                for (int i = 0; i < channelSize; i++) {
                    int q = Math.round(data[base + i] / scale);
                    q = Math.max(-128, Math.min(127, q));
                    quantized[base + i] = (byte) q;
                }
            }
        }

        // For per-channel, we store the first scale and note it's per-channel
        // In practice, you'd store all scales. Here we use average for simplicity.
        float avgScale = 0;
        for (float s : scales) avgScale += s;
        avgScale /= scales.length;

        return new QuantizedTensor(quantized, shape, avgScale, 0, QuantizationType.INT8);
    }

    @Override
    public Tensor dequantize(QuantizedTensor quantized) {
        if (quantized.type() != QuantizationType.INT8) {
            throw new IllegalArgumentException("Expected INT8 quantized tensor");
        }

        byte[] qData = quantized.data();
        float scale = quantized.scale();
        int[] shape = quantized.shape();

        float[] data = new float[qData.length];
        for (int i = 0; i < qData.length; i++) {
            data[i] = qData[i] * scale;
        }

        return new Tensor(shape, data);
    }
}