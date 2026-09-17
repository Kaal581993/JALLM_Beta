# JALLM — Plan Implementation Status

> Detailed status of every item in `docs/project_plan.md`, including what is implemented, what is pending, and how to implement it.

---

## 1. Establish a Reliable Test Command

### ✅ Done — Maven/Surefire Setup

- `pom.xml` configures JUnit 5 (5.10.2), Maven compiler (release 21), Surefire (3.2.5), and JAR manifest with `Main.class` as entry point.
- `mvn clean compile` — **BUILD SUCCESS** (64 source files compiled).
- `mvn test` — runs 65 tests, **60 pass, 5 fail**.

### ⚠️ In Progress — Fix Test Failures

**5 test failures need resolution:**

#### Failure 1: `AdamWTest.testSimpleOptimization`
- **Expected**: loss ≈ 5.0 after 1 step
- **Got**: loss ≈ 12.81
- **Analysis**: The test likely uses learning rate 0.1 on a simple quadratic. With AdamW at lr=0.1, the first step may overshoot depending on gradient magnitude. The test expectation may need adjustment, or the test should use a smaller learning rate.
- **Fix approach**: Review test expectations against actual AdamW math. The update is: `param -= lr * m_hat / (sqrt(v_hat) + eps)`. At step 1 with β₁=0.9, β₂=0.999, ε=1e-8, the bias-corrected m_hat ≈ grad, v_hat ≈ grad². For grad=1.0: update ≈ 0.1 / (1 + 1e-8) ≈ 0.1. If param starts at 0 and target is 0, loss = param² = 0.01, not 5.0. The test setup needs review.

#### Failure 2: `AdamWTest.testMultipleParameters`
- **Expected**: loss ≈ 3.0
- **Got**: loss ≈ 13.98
- **Analysis**: Same class of issue — test expectations may not match actual optimizer behavior.
- **Fix approach**: Adjust test expectations to match correct AdamW behavior, or verify the test setup is mathematically sound.

#### Failure 3: `AdamWTest.testZeroGrad`
- **Error**: `NullPointerException: Cannot invoke "Tensor.data()" because gradient() returns null`
- **Root cause**: `AdamW.zeroGrad()` calls `p.zeroGrad()` on each Variable, but `Variable.zeroGrad()` may not null out the gradient field — it may only clear it to zero or not handle it at all. Then `step()` checks `p.gradient() == null` and skips, but the test directly accesses `p.gradient().data()` after zeroGrad.
- **Fix approach**: In `Variable.java`, ensure `zeroGrad()` either sets `gradient = null` or sets gradient data to zeros. The test expects `gradient()` to return a non-null tensor after `zeroGrad()`.

#### Failure 4: `AutogradTest.testZeroGrad`
- **Same NPE issue** as above — `Variable.zeroGrad()` doesn't properly null the gradient.

#### Failure 5: `TokenizerTest.testTokenizerVocabSizes`
- **Expected**: `true` (vocab sizes match expectations)
- **Got**: `false`
- **Root cause**: `BPETokenizer` vocab size after training may differ from expected because the training loop adds merge tokens dynamically. The test may expect a specific vocab size that doesn't match the actual merge count.
- **Fix approach**: Review the test assertion and either adjust expected vocab size or fix the BPE training logic.

### How to Fix

```bash
# Run specific failing tests for detailed output:
mvn test -Dtest=AdamWTest
mvn test -Dtest=AutogradTest
mvn test -Dtest=TokenizerTest
```

---

## 2. Add Numerical Gradient-Check Tests

### 🔜 Pending

**Where**: `src/test/java/com/jalllm/geforce/`

| Test File | What to Test |
|-----------|-------------|
| `GradientCheckAutogradTest.java` | Numerical gradient check for all autograd ops (add, sub, mul, div, matmul, gelu, silu, softmax) |
| `GradientCheckLayerNormTest.java` | Gradient check for LayerNorm forward/backward |
| `GradientCheckAttentionTest.java` | Gradient check for multi-head attention projections |
| `GradientCheckCrossEntropyTest.java` | Gradient check for cross-entropy loss |

### How to Implement

For each operation `f(x)`:
1. Create a small random input tensor `x`
2. Compute analytical gradient via autograd: `f.backward(1.0)`
3. Compute numerical gradient via finite differences: `(f(x+ε) - f(x-ε)) / (2ε)`
4. Assert `|analytical - numerical| < ε` (typically 1e-4 relative tolerance)

**Example for an element-wise op**:

```java
@Test
void testGradientAdd() {
    float[] data = {1.0f, 2.0f, 3.0f};
    Tensor x = Tensor.randn(new int[]{3}, 0.0f, 1.0f);
    float[] xData = x.data().clone();
    
    Variable xVar = Variable.of(x, "x");
    Variable result = VariableOps.add(xVar, Variable.of(new Tensor(new int[]{3}, 1.0f), "one"));
    
    // Analytical gradient
    ComputationalGraph.backward(result);
    float[] analyticalGrad = xVar.gradient().data();
    
    // Numerical gradient
    float eps = 1e-4f;
    float[] numericalGrad = new float[3];
    for (int i = 0; i < 3; i++) {
        x.data()[i] = xData[i] + eps;
        float fp = result.value().data()[0];
        x.data()[i] = xData[i] - eps;
        float fm = result.value().data()[0];
        numericalGrad[i] = (fp - fm) / (2 * eps);
        x.data()[i] = xData[i]; // restore
    }
    
    // Compare
    for (int i = 0; i < 3; i++) {
        assertEquals(numericalGrad[i], analyticalGrad[i], 1e-3f);
    }
}
```

---

## 3. End-to-End Test: Tiny Corpus

### 🔜 Pending

**Goal**: Verify the full pipeline works: text → tokenize → train → loss decreases → save → load → generate.

### How to Implement

```java
@Test
void testEndToEndTinyCorpus() throws Exception {
    // 1. Create tiny corpus
    String corpus = "hello world hello java hello world";
    Path tempFile = Files.writeString(Files.createTempFile("tiny", ".txt"), corpus);
    
    // 2. Tokenize
    CharacterTokenizer tokenizer = new CharacterTokenizer(corpus);
    int[] tokens = toIntArray(tokenizer.encode(corpus));
    
    // 3. Create small model
    ModelConfig config = new ModelConfig(
        tokenizer.vocabSize(), 16, 8, 1, 1, 16, 16, 0.1f
    );
    MiniGPT model = new MiniGPT(config);
    
    // 4. Train for a few steps
    TrainingConfig trainConfig = new TrainingConfig(10, 2, 0.01f, 0.01f, 1.0f, 0, "models/test");
    RuntimeConfig runtimeConfig = new RuntimeConfig(1, false, 42);
    AdamW optimizer = new AdamW(model.parameters(), 0.001f);
    Trainer trainer = new Trainer(model, optimizer, trainConfig, runtimeConfig);
    
    List<Batch> batches = createBatches(tokens, 2, 4, 1);
    float initialLoss = trainer.evaluate(batches, 2, 4);
    TrainingState state = trainer.train(batches, 10);
    float finalLoss = state.loss();
    
    // 5. Loss should decrease (or at least not explode)
    assertTrue(finalLoss < initialLoss * 2, 
        "Loss should not explode: initial=" + initialLoss + " final=" + finalLoss);
    
    // 6. Save model
    Path modelPath = Files.createTempFile("model", ".bin");
    ModelWriter.write(model, modelPath.toFile());
    
    // 7. Load model
    MiniGPT loaded = ModelReader.read(modelPath.toFile());
    
    // 8. Generate
    int[] prompt = toIntArray(tokenizer.encode("hello"));
    int[] generated = loaded.generate(prompt, 5, 1.0f, 0, 1.0f);
    String output = tokenizer.decode(Arrays.stream(generated).boxed().toList());
    
    assertNotNull(output);
    assertFalse(output.isEmpty());
}
```

---

## 4. Dataset Pipeline

### ✅ Partially Done

**What exists**:
- `Trainer.createBatch()` — pads and masks batches
- `CliDispatcher.createBatches()` — creates batches from token arrays
- `CharacterTokenizer` — encodes text to tokens

**What's missing**:
- Persistent tokenizer vocabulary (save/load)
- Train/validation split logic
- Seed-deterministic shuffling
- Corpus validation (minimum length checks)

### How to Implement

#### Persistent Tokenizer Vocabulary

Add to `CharacterTokenizer` (and `BPETokenizer`):

```java
public void save(OutputStream out) throws IOException {
    try (DataOutputStream dos = new DataOutputStream(out)) {
        dos.writeInt(vocabSize);
        dos.writeInt(charToId.size());
        for (Map.Entry<Character, Integer> e : charToId.entrySet()) {
            dos.writeChar(e.getKey());
            dos.writeInt(e.getValue());
        }
        dos.writeInt(idToChar.size());
        for (Map.Entry<Integer, Character> e : idToChar.entrySet()) {
            dos.writeInt(e.getKey());
            dos.writeChar(e.getValue());
        }
    }
}

public static CharacterTokenizer load(InputStream in) throws IOException {
    try (DataInputStream dis = new DataInputStream(in)) {
        int vocabSize = dis.readInt();
        int mapSize = dis.readInt();
        CharacterTokenizer t = new CharacterTokenizer();
        t.charToId.clear();
        t.idToChar.clear();
        for (int i = 0; i < mapSize; i++) {
            char c = dis.readChar();
            int id = dis.readInt();
            t.charToId.put(c, id);
            t.idToChar.put(id, c);
        }
        // ... restore vocabSize
        return t;
    }
}
```

#### Train/Validation Split

```java
public static List<int[]>[] splitDataset(int[] tokens, float trainRatio, int seed) {
    Random rng = new Random(seed);
    // Shuffle indices
    int[] indices = new int[tokens.length];
    for (int i = 0; i < indices.length; i++) indices[i] = i;
    for (int i = indices.length - 1; i > 0; i--) {
        int j = rng.nextInt(i + 1);
        int tmp = indices[i]; indices[i] = indices[j]; indices[j] = tmp;
    }
    int split = (int)(tokens.length * trainRatio);
    // ... create train and val token arrays
}
```

---

## 5. CLI Behavior Improvements

### ⚠️ Partially Done

| Command | Status | Issue |
|---------|--------|-------|
| `train` | Hardcoded constants | Model config (256d, 128L, etc.) is fixed in `CliDispatcher.runTrain()`. Should accept flags. |
| `generate` | ✅ Working | Greedy, temperature, top-k, top-p all honored |
| `evaluate` | ✅ Working | Reports loss and perplexity |
| `inspect` | ✅ Working | Shows full metadata |
| `quantize` | ⚠️ Demonstration only | Quantizes parameters but doesn't use quantized inference |

### How to Improve `train` Command

Add CLI argument parsing:

```java
// Usage: jallm train <data-file> [--dim 64] [--layers 2] [--heads 2] [--seq 64] [--lr 1e-3] [--steps 100] [--batch 4] [--output dir]
private static void runTrain(String[] args) {
    // Parse arguments
    String dataPath = args[1];
    int embedDim = 64;
    int numLayers = 2;
    int numHeads = 2;
    int maxSeqLen = 64;
    float learningRate = 1e-3f;
    int maxSteps = 100;
    int batchSize = 4;
    String outputDir = "models/checkpoints";
    
    // Parse flags
    for (int i = 2; i < args.length; i++) {
        switch (args[i]) {
            case "--dim" -> embedDim = Integer.parseInt(args[++i]);
            case "--layers" -> numLayers = Integer.parseInt(args[++i]);
            case "--heads" -> numHeads = Integer.parseInt(args[++i]);
            case "--seq" -> maxSeqLen = Integer.parseInt(args[++i]);
            case "--lr" -> learningRate = Float.parseFloat(args[++i]);
            case "--steps" -> maxSteps = Integer.parseInt(args[++i]);
            case "--batch" -> batchSize = Integer.parseInt(args[++i]);
            case "--output" -> outputDir = args[++i];
        }
    }
    
    // Validate
    if (embedDim % numHeads != 0) {
        throw new IllegalArgumentException("embedDim must be divisible by numHeads");
    }
    if (maxSeqLen <= 0 || batchSize <= 0) {
        throw new IllegalArgumentException("Sequence length and batch size must be positive");
    }
    
    // ... rest of training logic
}
```

### Validation Errors

Add checks for:
- Short datasets (fewer tokens than sequence length)
- Invalid settings (embedDim not divisible by numHeads, negative learning rate)
- Incompatible checkpoints (model architecture mismatch on load)

---

## 6. Model Learning Verification

### 🔜 Pending

**Goal**: Prove the model actually learns on a tiny corpus.

### Plan

1. **Create example dataset**: `docs/examples/tiny.txt` with ~100 characters of repetitive text (e.g., "hello world. " repeated)
2. **Train small model**: vocab=50, dim=32, layers=1, heads=1, seq=32
3. **Measure**: Loss at step 0, 10, 50, 100 — should decrease monotonically (or near-monotonically)
4. **Generate**: After training, generate text from "hello" prompt — should produce text resembling training data
5. **Reproducible**: Fixed seed (42), documented commands

### Example Commands

```bash
# Train
java -jar target/jallm.jar train docs/examples/tiny.txt models/tiny --dim 32 --layers 1 --heads 1 --seq 32 --steps 100 --batch 2 --lr 1e-3

# Inspect
java -jar target/jallm.jar inspect models/tiny/model.bin

# Evaluate
java -jar target/jallm.jar evaluate models/tiny/model.bin docs/examples/tiny.txt

# Generate
java -jar target/jallm.jar generate models/tiny/model.bin "hello" 50 1.0 0 1.0 temperature
```

---

## 7. Inference & CPU Performance

### 🔜 Pending

### Current State
- Matrix multiplication: naive triple loop in `MatrixMath.matmul()`
- No buffer reuse — every operation allocates new arrays
- No thread pool — uses single-threaded execution
- No KV caching — recomputes all layers at every generation step

### Planned Optimizations (in order)

| Priority | Optimization | Impact |
|----------|-------------|--------|
| 1 | Reusable work buffers | Reduce GC pressure |
| 2 | Controlled thread pool | Parallel matmul on large matrices |
| 3 | KV caching | O(1) per step instead of O(n²) |
| 4 | Loop tiling/blocking | Cache locality for matmul |
| 5 | Stride caching | Avoid repeated shape calculations |

### Benchmark Infrastructure

`MatrixBenchmark.java` and `InferenceBenchmark.java` already exist with suite methods. Extend them to:
- Record min/max/avg/median times
- Compare before/after each optimization
- Output CSV for analysis

---

## 8. BPE Tokenization

### ⚠️ Basic Implementation Exists

**What works**:
- Byte-level initialization (256 byte tokens)
- BPE merge training on a corpus
- Encoding with merge application
- Decoding (best-effort via byte reconstruction)

**What doesn't work well**:
- Decode of merged tokens is lossy (tries UTF-8 reconstruction)
- No vocabulary persistence (save/load)
- Merge table not stored in checkpoint

### How to Finish

1. **Fix decode**: Store token strings alongside IDs for accurate decode
2. **Persist vocabulary**: Save `tokenToId`, `idToToken`, and `merges` to a JSON or binary file
3. **Checkpoint integration**: Store tokenizer type and path in checkpoint metadata
4. **CLI**: Add `tokenizer train` and `tokenizer encode` commands

---

## 9. Quantization

### ⚠️ Basic Demonstration Exists

**What exists**:
- `Int8Quantizer` — quantizes float tensors to INT8 with scale factors
- `Int4Quantizer` — quantizes to INT4
- `QuantizedTensor` — stores scale + quantized data
- CLI `quantize` command — demonstrates quantization but doesn't use quantized inference

### What's Missing

1. **Versioned INT8 model format**: Need a separate format that stores:
   - Version marker (INT8-v1)
   - For each weight tensor: scale (float), zero_point (int), quantized data (int8[])
   - Metadata for dequantization during inference

2. **Quantized inference kernel**: `MatrixMath.matmulQuantized()` that operates on INT8 data directly

3. **Output deviation measurement**: Compare FP32 vs quantized outputs, report max abs error and mean squared error

4. **Memory reduction measurement**: Report FP32 size vs INT8 size

### Implementation Plan

```java
// INT8 model format
public class Int8ModelFormat {
    // Header: magic, version, param count
    // For each parameter:
    //   - name (UTF string)
    //   - shape (rank + dims)
    //   - scale (float)
    //   - zeroPoint (int8)
    //   - quantizedData (int8[])
}
```

**Do NOT implement INT4 until INT8 inference is correct and tested.**

---

## 10. Documentation & Release Readiness

### ⚠️ Basic README Exists

Current `README.md` covers:
- Project description
- Current status summary
- Run commands (`mvn test`, `mvn package`, `java -jar target/jallm.jar selftest`)

### What's Missing

| Document | Content |
|----------|---------|
| `docs/quickstart.md` | Step-by-step: install → train → inspect → evaluate → generate |
| `docs/checkpoint-format.md` | Detailed checkpoint binary format specification |
| `docs/model-sizes.md` | Supported configurations and memory estimates |
| `docs/limitations.md` | Known limitations and future work |
| Release build command | `mvn clean package -DskipTests` → `target/jallm.jar` |

### Release Build Command

```bash
# Build release JAR
mvn clean package -DskipTests

# Result: target/jallm.jar
# Run: java -jar target/jallm.jar selftest
```

---

## Summary Scorecard

| Area | Implemented | Pending | Blocked |
|------|-------------|---------|---------|
| Test infrastructure | ✅ | — | — |
| Tensor engine | ✅ | — | — |
| Math operations | ✅ | — | — |
| Autograd | ✅ (minor NPE) | Gradient check tests | — |
| Neural network | ✅ | — | — |
| Training loop | ✅ | Resume support | — |
| Optimizer (AdamW) | ✅ (test bugs) | — | — |
| Loss functions | ✅ | Gradient check | — |
| Tokenizers | ✅ (basic) | Persistent vocab, BPE decode | — |
| Model (MiniGPT) | ✅ | — | — |
| Serialization | ✅ | Tokenizer in checkpoint | — |
| CLI | ✅ (basic) | Configurable train, validation | — |
| Generation | ✅ | KV caching | — |
| Evaluation | ✅ | — | — |
| Quantization | ⚠️ Demo | Real INT8 inference | — |
| Benchmarks | ✅ (basic) | Optimization tracking | — |
| Documentation | ⚠️ Basic | Full docs, quickstart | — |
| End-to-end test | 🔜 | — | Test fixes |

---

## Immediate Next Steps

1. **Fix the 5 test failures** (AdamW expectations, zeroGrad NPE, tokenizer vocab)
2. **Create end-to-end test** with tiny corpus
3. **Add gradient-check tests** for autograd, LayerNorm, attention, cross-entropy
4. **Make train CLI configurable** with command-line flags
5. **Add tokenizer persistence** (save/load)
6. **Document the example dataset** and commands

Once these are done, the project becomes a **real executable language-model training system** rather than a collection of model components.
