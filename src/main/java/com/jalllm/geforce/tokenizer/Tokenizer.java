package com.jalllm.geforce.tokenizer;

import java.util.List;

/**
 * Base interface for tokenizers.
 */
public interface Tokenizer {

    /**
     * Encodes a string into token IDs.
     */
    List<Integer> encode(String text);

    /**
     * Decodes token IDs back into a string.
     */
    String decode(List<Integer> tokenIds);

    /**
     * Returns the vocabulary size.
     */
    int vocabSize();

    /**
     * Returns the token ID for the padding token.
     */
    int padTokenId();

    /**
     * Returns the token ID for the unknown token.
     */
    int unkTokenId();

    /**
     * Returns the token ID for the beginning-of-sequence token.
     */
    int bosTokenId();

    /**
     * Returns the token ID for the end-of-sequence token.
     */
    int eosTokenId();
}