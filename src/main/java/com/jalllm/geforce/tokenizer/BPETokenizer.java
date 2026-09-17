package com.jalllm.geforce.tokenizer;

import java.util.*;
import java.util.regex.Pattern;

/**
 * Byte Pair Encoding (BPE) Tokenizer.
 * Learns subword units from a corpus.
 */
public final class BPETokenizer implements Tokenizer {

    private final Map<String, Integer> tokenToId;
    private final Map<Integer, String> idToToken;
    private final Map<String, Integer> merges; // pair -> merge index
    private final int vocabSize;
    private final int padTokenId;
    private final int unkTokenId;
    private final int bosTokenId;
    private final int eosTokenId;
    private final Pattern wordPattern;

    public BPETokenizer(int vocabSize) {
        this.vocabSize = vocabSize;
        this.tokenToId = new HashMap<>();
        this.idToToken = new HashMap<>();
        this.merges = new LinkedHashMap<>(); // Preserve insertion order
        this.padTokenId = 0;
        this.unkTokenId = 1;
        this.bosTokenId = 2;
        this.eosTokenId = 3;

        // Special tokens
        addToken("<pad>", padTokenId);
        addToken("?</", unkTokenId); // Using a unique string
        addToken("<bos>", bosTokenId);
        addToken("<eos>", eosTokenId);

        // Initialize with byte-level tokens (0-255)
        for (int i = 0; i < 256; i++) {
            String byteToken = String.format("<0x%02X>", i);
            int id = 4 + i;
            addToken(byteToken, id);
        }

        // Word pattern: split on whitespace, keep punctuation with words
        this.wordPattern = Pattern.compile("\\S+");
    }

    private void addToken(String token, int id) {
        tokenToId.put(token, id);
        idToToken.put(id, token);
    }

    /**
     * Trains the BPE tokenizer on a corpus.
     */
    public void train(List<String> corpus) {
        // Count word frequencies
        Map<String, Integer> wordFreqs = new HashMap<>();
        for (String text : corpus) {
            for (String word : splitWords(text)) {
                // Convert word to byte tokens
                List<String> byteTokens = wordToByteTokens(word);
                String key = String.join(" ", byteTokens);
                wordFreqs.merge(key, 1, Integer::sum);
            }
        }

        // Perform BPE merges
        int numMerges = vocabSize - tokenToId.size();
        for (int i = 0; i < numMerges; i++) {
            // Find most frequent pair
            Map<String, Integer> pairFreqs = new HashMap<>();
            for (Map.Entry<String, Integer> entry : wordFreqs.entrySet()) {
                String[] tokens = entry.getKey().split(" ");
                int freq = entry.getValue();
                for (int j = 0; j < tokens.length - 1; j++) {
                    String pair = tokens[j] + " " + tokens[j + 1];
                    pairFreqs.merge(pair, freq, Integer::sum);
                }
            }

            if (pairFreqs.isEmpty()) break;

            // Get most frequent pair
            String bestPair = pairFreqs.entrySet().stream()
                    .max(Map.Entry.comparingByValue())
                    .map(Map.Entry::getKey)
                    .orElse(null);

            if (bestPair == null) break;

            // Add merge
            String[] pairParts = bestPair.split(" ");
            String merged = pairParts[0] + pairParts[1];
            int newId = tokenToId.size();
            addToken(merged, newId);
            merges.put(bestPair, newId);

            // Apply merge to word frequencies
            Map<String, Integer> newWordFreqs = new HashMap<>();
            for (Map.Entry<String, Integer> entry : wordFreqs.entrySet()) {
                String word = entry.getKey();
                int freq = entry.getValue();
                String mergedWord = word.replace(" " + bestPair + " ", " " + merged + " ")
                        .replace(bestPair + " ", merged + " ")
                        .replace(" " + bestPair, " " + merged);
                newWordFreqs.merge(mergedWord, freq, Integer::sum);
            }
            wordFreqs = newWordFreqs;
        }
    }

    private List<String> splitWords(String text) {
        List<String> words = new ArrayList<>();
        java.util.regex.Matcher matcher = wordPattern.matcher(text);
        while (matcher.find()) {
            words.add(matcher.group());
        }
        return words;
    }

    private List<String> wordToByteTokens(String word) {
        List<String> tokens = new ArrayList<>();
        for (byte b : word.getBytes(java.nio.charset.StandardCharsets.UTF_8)) {
            int unsigned = b & 0xFF;
            tokens.add(String.format("<0x%02X>", unsigned));
        }
        return tokens;
    }

    @Override
    public List<Integer> encode(String text) {
        List<Integer> ids = new ArrayList<>();
        ids.add(bosTokenId);

        for (String word : splitWords(text)) {
            List<String> byteTokens = wordToByteTokens(word);
            // Apply merges
            List<String> tokens = applyMerges(byteTokens);
            for (String token : tokens) {
                ids.add(tokenToId.getOrDefault(token, unkTokenId));
            }
        }

        ids.add(eosTokenId);
        return ids;
    }

    private List<String> applyMerges(List<String> tokens) {
        boolean changed = true;
        while (changed && tokens.size() > 1) {
            changed = false;
            // Find the earliest merge that applies
            int bestIdx = -1;
            int bestMergeOrder = Integer.MAX_VALUE;

            for (int i = 0; i < tokens.size() - 1; i++) {
                String pair = tokens.get(i) + " " + tokens.get(i + 1);
                Integer mergeOrder = merges.get(pair);
                if (mergeOrder != null && mergeOrder < bestMergeOrder) {
                    bestMergeOrder = mergeOrder;
                    bestIdx = i;
                }
            }

            if (bestIdx >= 0) {
                String merged = tokens.get(bestIdx) + tokens.get(bestIdx + 1);
                tokens.set(bestIdx, merged);
                tokens.remove(bestIdx + 1);
                changed = true;
            }
        }
        return tokens;
    }

    @Override
    public String decode(List<Integer> tokenIds) {
        StringBuilder sb = new StringBuilder();
        List<Byte> bytes = new ArrayList<>();

        for (int id : tokenIds) {
            if (id == bosTokenId || id == eosTokenId || id == padTokenId) {
                continue;
            }
            String token = idToToken.get(id);
            if (token == null) {
                token = idToToken.get(unkTokenId);
            }

            // Parse byte tokens
            if (token.startsWith("<0x") && token.endsWith(">")) {
                String hex = token.substring(3, token.length() - 1);
                try {
                    int byteVal = Integer.parseInt(hex, 16);
                    bytes.add((byte) byteVal);
                } catch (NumberFormatException e) {
                    // Invalid byte token, skip
                }
            } else if (token.length() > 1) {
                // Merged token - need to decompose
                // For simplicity, we'll try to decode as UTF-8 bytes
                for (byte b : token.getBytes(java.nio.charset.StandardCharsets.UTF_8)) {
                    bytes.add(b);
                }
            }
        }

        try {
            return new String(toByteArray(bytes), java.nio.charset.StandardCharsets.UTF_8);
        } catch (Exception e) {
            return "[decode error]";
        }
    }

    private byte[] toByteArray(List<Byte> bytes) {
        byte[] arr = new byte[bytes.size()];
        for (int i = 0; i < bytes.size(); i++) {
            arr[i] = bytes.get(i);
        }
        return arr;
    }

    @Override
    public int vocabSize() {
        return tokenToId.size();
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

    public Map<String, Integer> getTokenToId() {
        return new HashMap<>(tokenToId);
    }

    public Map<Integer, String> getIdToToken() {
        return new HashMap<>(idToToken);
    }

    public Map<String, Integer> getMerges() {
        return new LinkedHashMap<>(merges);
    }
}