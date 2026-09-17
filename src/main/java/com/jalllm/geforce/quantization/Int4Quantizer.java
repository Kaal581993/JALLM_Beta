package com.jalllm.geforce.quantization;

import com.jalllm.geforce.tensor.Tensor;

/**
 * INT4 Quantization (symmetric).
 * Packs two 4-bit values into one byte.
 * q = round(x / scale) clamped to [-8, 7]
 * x = q * scale
 */
public final class Int4Quantizer implements Quantizer {

    private final boolean perChannel;
    private final int channelAxis;

    public Int4Quantizer() {
        this(false, -1);
    }

    public Int4Quantizer(boolean perChannel, int channelAxis) {
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

        // Symmetric quantization: scale = max(abs(min), abs(max)) / 7
        float maxAbs = Math.max(Math.abs(min), Math.abs(max));
        float scale = maxAbs / 7.0f;
        if (scale == 0) scale = 1.0f;

        // Quantize and pack (2 values per byte)
        int packedSize = (data.length + 1) / 2;
        byte[] quantized = new byte[packedSize];

        for (int i = 0; i < data.length; i += 2) {
            int q1 = Math.round(data[i] / scale);
            q1 = Math.max(-8, Math.min(7, q1));

            int q2 = 0;
            if (i + 1 < data.length) {
                q2 = Math.round(data[i + 1] / scale);
                q2 = Math.max(-8, Math.min(7, q2));
            }

            // Pack: high 4 bits = q1, low 4 bits = q2
            // Convert to unsigned 4-bit representation (0-15)
            int uq1 = q1 + 8;
            int uq2 = q2 + 8;
            quantized[i / 2] = (byte) ((uq1 << 4) | uq2);
        }

        return new QuantizedTensor(quantized, shape, scale, 8, QuantizationType.INT4);
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
        int packedSize = (data.length + 1) / 2;
        byte[] quantized = new byte[packedSize];

        for (int c = 0; c < numChannels; c++) {
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
            float scale = maxAbs / 7.0f;
            if (scale == 0) scale = 1.0f;
            scales[c] = scale;

            // Quantize this channel
            for (int o = 0; o < outerSize; o++) {
                int base = o * numChannels * channelSize + c * channelSize;
                for (int i = 0; i < channelSize; i += 2) {
                    int q1 = Math.round(data[base + i] / scale);
                    q1 = Math.max(-8, Math.min(7, q1));

                    int q2 = 0;
                    if (i + 1 < channelSize) {
                        q2 = Math.round(data[base + i + 1] / scale);
                        q2 = Math.max(-8, Math.min(7, q2));
                    }

                    int uq1 = q1 + 8;
                    int uq2 = q2 + 8;
                    int packedIdx = (base + i) / 2;
                    quantized[packedIdx] = (byte) ((uq1 << 4) | uq2);
                }
            }
        }

        float avgScale = 0;
        for (float s : scales) avgScale += s;
        avgScale /= scales.length;

        return new QuantizedTensor(quantized, shape, avgScale, 8, QuantizationType.INT4);
    }

    @Override
    public Tensor dequantize(QuantizedTensor quantized) {
        if (quantized.type() != QuantizationType.INT4) {
            throw new IllegalArgumentException("Expected INT4 quantized tensor");
        }

        byte[] qData = quantized.data();
        float scale = quantized.scale();
        int zeroPoint = quantized.zeroPoint(); // Should be 8 for symmetric
        int[] shape = quantized.shape();

        int totalElements = 1;
        for (int dim : shape) totalElements *= dim;

        float[] data = new float[totalElements];

        for (int i = 0; i < qData.length; i++) {
            byte packed = qData[i];

            // Unpack high 4 bits
            int uq1 = (packed >> 4) & 0xF;
            int q1 = uq1 - zeroPoint;
            data[i * 2] = q1 * scale;

            // Unpack low 4 bits
            if (i * 2 + 1 < totalElements) {
                int uq2 = packed & 0xF;
                int q2 = uq2 - zeroPoint;
                data[i * 2 + 1] = q2 * scale;
            }
        }

        return new Tensor(shape, data);
    }
}