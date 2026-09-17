package com.jalllm.geforce.tokenizer;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Character-level tokenizer.
 * Each character is a token. Simple but vocabulary size equals character set size.
 */
public final class CharacterTokenizer implements Tokenizer {

    private final Map<Character, Integer> charToId;
    private final Map<Integer, Character> idToChar;
    private int vocabSize;
    private final int padTokenId;
    private final int unkTokenId;
    private final int bosTokenId;
    private final int eosTokenId;

    public CharacterTokenizer() {
        this.charToId = new HashMap<>();
        this.idToChar = new HashMap<>();

        // Special tokens
        this.padTokenId = 0;
        this.unkTokenId = 1;
        this.bosTokenId = 2;
        this.eosTokenId = 3;

        charToId.put('<', padTokenId);   // <pad>
        charToId.put('?', unkTokenId);   // <unk>
        charToId.put('[', bosTokenId);   // <bos>
        charToId.put(']', eosTokenId);   // <eos>

        idToChar.put(padTokenId, '<');
        idToChar.put(unkTokenId, '?');
        idToChar.put(bosTokenId, '[');
        idToChar.put(eosTokenId, ']');

        // Add printable ASCII characters (32-126)
        int nextId = 4;
        for (int c = 32; c <= 126; c++) {
            char ch = (char) c;
            if (!charToId.containsKey(ch)) {
                charToId.put(ch, nextId);
                idToChar.put(nextId, ch);
                nextId++;
            }
        }

        // Add common Unicode characters
        for (char ch : new char[]{'\n', '\t', '\r'}) {
            if (!charToId.containsKey(ch)) {
                charToId.put(ch, nextId);
                idToChar.put(nextId, ch);
                nextId++;
            }
        }

        this.vocabSize = nextId;
    }

    public CharacterTokenizer(String corpus) {
        this();
        // Build vocabulary from corpus
        for (char ch : corpus.toCharArray()) {
            if (!charToId.containsKey(ch)) {
                charToId.put(ch, vocabSize);
                idToChar.put(vocabSize, ch);
                vocabSize++;
            }
        }
    }

    @Override
    public List<Integer> encode(String text) {
        List<Integer> ids = new ArrayList<>();
        ids.add(bosTokenId);
        for (char ch : text.toCharArray()) {
            ids.add(charToId.getOrDefault(ch, unkTokenId));
        }
        ids.add(eosTokenId);
        return ids;
    }

    @Override
    public String decode(List<Integer> tokenIds) {
        StringBuilder sb = new StringBuilder();
        for (int id : tokenIds) {
            if (id == bosTokenId || id == eosTokenId || id == padTokenId) {
                continue; // Skip special tokens in decoding
            }
            Character ch = idToChar.get(id);
            if (ch != null) {
                sb.append(ch);
            }
        }
        return sb.toString();
    }

    @Override
    public int vocabSize() {
        return vocabSize;
    }

    @Override
    public int padTokenId() {
        return padTokenId;
    }

    @Override
    public int unkTokenId() {
        return unkTokenId;
    }

    @Override
    public int bosTokenId() {
        return bosTokenId;
    }

    @Override
    public int eosTokenId() {
        return eosTokenId;
    }

    public Map<Character, Integer> getCharToId() {
        return new HashMap<>(charToId);
    }

    public Map<Integer, Character> getIdToChar() {
        return new HashMap<>(idToChar);
    }
}
