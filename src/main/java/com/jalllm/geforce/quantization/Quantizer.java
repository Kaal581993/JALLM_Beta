package com.jalllm.geforce.quantization;

import com.jalllm.geforce.tensor.Tensor;

/**
 * Base interface for quantizers.
 */
public interface Quantizer {

    /**
     * Quantizes a float tensor to a lower precision representation.
     *
     * @param tensor Input float tensor
     * @return Quantized tensor with scale and zero-point info
     */
    QuantizedTensor quantize(Tensor tensor);

    /**
     * Dequantizes a quantized tensor back to float.
     *
     * @param quantized Quantized tensor
     * @return Dequantized float tensor
     */
    Tensor dequantize(QuantizedTensor quantized);
}