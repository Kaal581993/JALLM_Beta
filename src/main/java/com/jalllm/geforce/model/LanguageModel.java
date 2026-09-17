package com.jalllm.geforce.model;

import com.jalllm.geforce.autograd.Variable;
import com.jalllm.geforce.autograd.VariableOps;
import com.jalllm.geforce.embedding.TokenEmbedding;
import com.jalllm.geforce.transformer.Transformer;
import com.jalllm.geforce.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;

/**
 * Base Language Model interface.
 */
public interface LanguageModel {

    /**
     * Forward pass with autograd (for training).
     * Input: token IDs [batch, seqLen]
     * Output: logits [batch, seqLen, vocabSize]
     */
    Variable forward(Variable tokenIds, Variable mask);

    /**
     * Forward pass without autograd (for inference).
     */
    Tensor forward(Tensor tokenIds, Tensor mask);

    /**
     * Returns all trainable parameters.
     */
    List<Variable> parameters();

    /**
     * Returns the vocabulary size.
     */
    int vocabSize();

    /**
     * Returns the embedding dimension.
     */
    int embedDim();

    /**
     * Returns the maximum sequence length.
     */
    int maxSeqLen();
}