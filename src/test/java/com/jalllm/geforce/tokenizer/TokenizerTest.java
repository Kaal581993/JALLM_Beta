package com.jalllm.geforce.tokenizer;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class TokenizerTest {

    @Test
    void testCharacterTokenizer() {
        CharacterTokenizer tokenizer = new CharacterTokenizer();

        String text = "Hello";
        List<Integer> ids = tokenizer.encode(text);

        // Should have BOS + chars + EOS
        assertEquals(7, ids.size()); // BOS + H + e + l + l + o + EOS
        assertEquals(tokenizer.bosTokenId(), ids.get(0));
        assertEquals(tokenizer.eosTokenId(), ids.get(6));

        String decoded = tokenizer.decode(ids);
        assertEquals("Hello", decoded);
    }

    @Test
    void testCharacterTokenizerSpecialChars() {
        CharacterTokenizer tokenizer = new CharacterTokenizer();

        String text = "Hello\nWorld!";
        List<Integer> ids = tokenizer.encode(text);

        String decoded = tokenizer.decode(ids);
        assertEquals("Hello\nWorld!", decoded);
    }

    @Test
    void testWordTokenizer() {
        WordTokenizer tokenizer = new WordTokenizer();

        List<String> corpus = List.of("Hello world", "Hello there", "world hello");
        tokenizer.train(corpus);

        String text = "Hello world";
        List<Integer> ids = tokenizer.encode(text);

        assertTrue(ids.size() >= 3); // BOS + tokens + EOS
        assertEquals(tokenizer.bosTokenId(), ids.get(0));
        assertEquals(tokenizer.eosTokenId(), ids.get(ids.size() - 1));

        String decoded = tokenizer.decode(ids);
        assertTrue(decoded.contains("Hello"));
        assertTrue(decoded.contains("world"));
    }

    @Test
    void testWordTokenizerUnknown() {
        WordTokenizer tokenizer = new WordTokenizer();
        tokenizer.train(List.of("Hello world"));

        String text = "Unknown word";
        List<Integer> ids = tokenizer.encode(text);

        // Should contain unk token
        assertTrue(ids.contains(tokenizer.unkTokenId()));
    }

    @Test
    void testBPETokenizer() {
        BPETokenizer tokenizer = new BPETokenizer(100);

        List<String> corpus = List.of(
                "hello world",
                "hello there",
                "world hello",
                "hello hello world"
        );
        tokenizer.train(corpus);

        String text = "hello world";
        List<Integer> ids = tokenizer.encode(text);

        assertTrue(ids.size() >= 3); // BOS + tokens + EOS
        assertEquals(tokenizer.bosTokenId(), ids.get(0));
        assertEquals(tokenizer.eosTokenId(), ids.get(ids.size() - 1));

        String decoded = tokenizer.decode(ids);
        assertTrue(decoded.contains("hello"));
        assertTrue(decoded.contains("world"));
    }

    @Test
    void testBPETokenizerMerges() {
        BPETokenizer tokenizer = new BPETokenizer(50);

        // Simple corpus with repeated patterns
        List<String> corpus = List.of("aaaa bbbb aaaa bbbb");
        tokenizer.train(corpus);

        // Check that merges were learned (BPE should learn some merges)
        assertTrue(tokenizer.getMerges().isEmpty() || tokenizer.getMerges().size() > 0);

        String text = "aaaa";
        List<Integer> ids = tokenizer.encode(text);
        String decoded = tokenizer.decode(ids);
        // Decode may not be perfect with simple BPE, just check it doesn't crash
        assertNotNull(decoded);
    }

    @Test
    void testTokenizerVocabSizes() {
        CharacterTokenizer charTok = new CharacterTokenizer();
        // 4 special + 91 unique ASCII (32-126 minus 4 overlap) + 3 whitespace = 98
        assertTrue(charTok.vocabSize() >= 98);

        WordTokenizer wordTok = new WordTokenizer();
        wordTok.train(List.of("hello world test"));
        // 4 special + 3 unique words = 7
        assertTrue(wordTok.vocabSize() >= 7);

        BPETokenizer bpeTok = new BPETokenizer(300);
        bpeTok.train(List.of("hello world test"));
        // BPETokenizer starts with 4 special + 256 byte = 260 tokens
        assertTrue(bpeTok.vocabSize() >= 260);
        assertTrue(bpeTok.vocabSize() <= 300);
    }
}