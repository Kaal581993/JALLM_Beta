package com.jalllm.geforce.tokenizer;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;

/**
 * Word-level tokenizer.
 * Splits text by whitespace and punctuation.
 */
public final class WordTokenizer implements Tokenizer {

    private final Map<String, Integer> wordToId;
    private final Map<Integer, String> idToWord;
    private final int vocabSize;
    private final int padTokenId;
    private final int unkTokenId;
    private final int bosTokenId;
    private final int eosTokenId;
    private final Pattern splitPattern;

    public WordTokenizer() {
        this.wordToId = new HashMap<>();
        this.idToWord = new HashMap<>();
        this.padTokenId = 0;
        this.unkTokenId = 1;
        this.bosTokenId = 2;
        this.eosTokenId = 3;

        // Special tokens
        addToken("<pad>", padTokenId);
        addToken("<unk>", unkTokenId);
        addToken("<bos>", bosTokenId);
        addToken("<eos>", eosTokenId);

        // Split on whitespace and keep punctuation as separate tokens
        this.splitPattern = Pattern.compile("(\\s+|[.,!?;:()\\[\\]{}\"'-])");
        this.vocabSize = 4; // Will be updated after training
    }

    public WordTokenizer(List<String> corpus) {
        this();
        train(corpus);
    }

    private void addToken(String token, int id) {
        wordToId.put(token, id);
        idToWord.put(id, token);
    }

    public void train(List<String> corpus) {
        int nextId = 4;
        for (String text : corpus) {
            List<String> tokens = tokenize(text);
            for (String token : tokens) {
                if (!wordToId.containsKey(token)) {
                    wordToId.put(token, nextId);
                    idToWord.put(nextId, token);
                    nextId++;
                }
            }
        }
    }

    private List<String> tokenize(String text) {
        List<String> tokens = new ArrayList<>();
        java.util.regex.Matcher matcher = splitPattern.matcher(text);
        int lastEnd = 0;
        while (matcher.find()) {
            if (matcher.start() > lastEnd) {
                String word = text.substring(lastEnd, matcher.start());
                if (!word.isEmpty()) tokens.add(word);
            }
            String delim = matcher.group();
            if (!delim.trim().isEmpty()) {
                tokens.add(delim);
            }
            lastEnd = matcher.end();
        }
        if (lastEnd < text.length()) {
            String word = text.substring(lastEnd);
            if (!word.isEmpty()) tokens.add(word);
        }
        return tokens;
    }

    @Override
    public List<Integer> encode(String text) {
        List<Integer> ids = new ArrayList<>();
        ids.add(bosTokenId);
        for (String token : tokenize(text)) {
            ids.add(wordToId.getOrDefault(token, unkTokenId));
        }
        ids.add(eosTokenId);
        return ids;
    }

    @Override
    public String decode(List<Integer> tokenIds) {
        StringBuilder sb = new StringBuilder();
        boolean first = true;
        for (int id : tokenIds) {
            if (id == bosTokenId || id == eosTokenId || id == padTokenId) {
                continue;
            }
            String token = idToWord.get(id);
            if (token != null) {
                if (!first && !isPunctuation(token)) {
                    sb.append(' ');
                }
                sb.append(token);
                first = false;
            }
        }
        return sb.toString();
    }

    private boolean isPunctuation(String token) {
        return token.matches("[.,!?;:()\\[\\]{}\"'-]");
    }

    @Override
    public int vocabSize() {
        return wordToId.size();
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

    public Map<String, Integer> getWordToId() {
        return new HashMap<>(wordToId);
    }

    public Map<Integer, String> getIdToWord() {
        return new HashMap<>(idToWord);
    }
}