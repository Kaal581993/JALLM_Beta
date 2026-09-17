# JALLM — Project Plan

> **JALLM** (Java LLM From Scratch): A decoder-only Transformer language model implemented and trained entirely in Java, with no external AI frameworks, APIs, or pretrained models.

---

## Table of Contents

1. [Vision & Goals](#vision--goals)
2. [Architecture Overview](#architecture-overview)
3. [Engineering Rules](#engineering-rules)
4. [Component Roadmap](#component-roadmap)
5. [Current Implementation Status](#current-implementation-status)
6. [Pending Work](#pending-work)
7. [Test Strategy](#test-strategy)
8. [Performance Roadmap](#performance-roadmap)
9. [CLI Command Reference](#cli-command-reference)
10. [Checkpoint Format](#checkpoint-format)
11. [Hardware & Memory Budget](#hardware--memory-budget)
12. [Release Criteria](#release-criteria)

---

## Vision & Goals

Build a **real, executable language-model training system** in Java that can:

- Read a text corpus, train a tokenizer, encode data, and split train/validation sets
- Train a small decoder-only Transformer from random initialization
- Save checkpoints with model parameters, optimizer state, and training metadata
- Reload checkpoints and continue training or generate text
- Generate text with greedy, temperature, top-k, and top-p sampling
- Evaluate loss and perplexity over a validation corpus
- Quantize models (INT8, eventually INT4) with measurable output deviation
- Run efficiently on CPU with controlled threading and reusable buffers

**"Completed and executable"** means: train from text → save checkpoint → reload → generate → evaluate → optionally quantize — all without misleading output.

---

## Architecture Overview

```
                    JALLM
                     │
        ┌────────────┴────────────┐
        │                         │
     Training                  Inference
        │                         │
        ▼                         ▼
   Training Data              Prompt
        │                         │
        ▼                         ▼
     Tokenizer                Tokenizer
        │                         │
        ▼                         ▼
     Token IDs                Token IDs
        │                         │
        └────────────┬────────────┘
                     ▼
                Embeddings
                     │
                     ▼
             Transformer × N
                     │
          ┌──────────┴──────────┐
          │                     │
     Self Attention         Feed Forward
          │                     │
          └──────────┬──────────┘
                     ▼
                 LayerNorm
                     │
                     ▼
                  Logits
                     │
                     ▼
                 Softmax
                     │
                     ▼
             Next Token
                     │
                     ▼
               Text Output
```

Training adds:

```
Prediction
    ↓
Cross Entropy Loss
    ↓
Backpropagation
    ↓
Gradients
    ↓
AdamW
    ↓
Updated Parameters
```

---

## Engineering Rules

### Rule 1 — No AI Frameworks

We do **not** use PyTorch, TensorFlow, DL4J, ONNX Runtime, Hugging Face, llama.cpp, or any pretrained model as our model. Ordinary Java libraries (JUnit, file I/O, collections) are acceptable. The neural-network implementation is entirely ours.

### Rule 2 — Bottom-Up Construction

Build in this order:

| Step | Component |
|------|-----------|
| 01 | Environment & project setup |
| 02 | Tensor engine (flat `float[]` storage) |
| 03 | Mathematical operations (element-wise, reductions) |
| 04 | Numerical stability (softmax, log-sum-exp) |
| 05 | Autograd (reverse-mode automatic differentiation) |
| 06 | Neural-network primitives (linear, embedding, layer norm) |
| 07 | Optimizers (AdamW) |
| 08 | Dataset engine (corpus → tokens → batches) |
| 09 | Tokenizer (character, word, BPE) |
| 10 | Embeddings |
| 11 | Attention (causal multi-head) |
| 12 | Transformer block |
| 13 | GPT-style model |
| 14 | Training loop |
| 15 | Text generation |
| 16 | Checkpointing |
| 17 | Quantization |
| 18 | CPU optimization |
| 19 | Final CLI |

### Rule 3 — Every Component Gets Tests

Each component is tested for: normal case, edge cases, error handling, and (where relevant) performance benchmark.

### Rule 4 — Reference Implementation First

```
Reference implementation → Correctness → Optimized implementation → Performance
```

Never optimize before proving mathematical correctness.

### Rule 5 — `float`, Not `double`

For eventual models with ~10M parameters:

| Precision | Memory (weights only) |
|-----------|----------------------|
| FP32 (`float`) | 10M × 4 = **40 MB** |
| FP64 (`double`) | 10M × 8 = **80 MB** |

Plus AdamW optimizer state (m, v), memory grows considerably. Design around contiguous primitive `float[]` arrays.

---

## Component Roadmap

### Phase 1 — Foundation (Milestone: End-to-End Tiny-Corpus Training)

| # | Task | Status |
|---|------|--------|
| 1 | Establish reliable test command (Maven/Surefire) | ✅ Done |
| 2 | Run all existing tests | ⚠️ 5 failures to fix |
| 3 | Add numerical gradient-check tests (autograd, LayerNorm, attention, cross-entropy) | 🔜 Pending |
| 4 | End-to-end test: tiny corpus → training → lower loss → save/load → generation | 🔜 Pending |
| 5 | Dataset pipeline: corpus → tokenizer → encode → split → batch → pad → mask | ✅ Partially done |
| 6 | Persistent tokenizer vocabulary | ⚠️ Needs improvement |
| 7 | Deterministic seed handling | ⚠️ Partial |
| 8 | Train/validation metrics | ✅ Done |
| 9 | Periodic checkpointing & resume | ⚠️ Needs improvement |
| 10 | Save/restore AdamW moments & training state | ⚠️ Needs improvement |
| 11 | CLI: train with configurable options | ⚠️ Hardcoded constants |
| 12 | CLI: generate with greedy/temperature/top-k/top-p | ✅ Done |
| 13 | CLI: evaluate (loss/perplexity) | ✅ Done |
| 14 | CLI: inspect (full metadata) | ✅ Done |
| 15 | Validation errors for short datasets, invalid settings | ⚠️ Needs improvement |

### Phase 2 — Model Learning Verification

| # | Task | Status |
|---|------|--------|
| 1 | Character-level corpus + very small model | 🔜 Pending |
| 2 | Confirm loss decreases | 🔜 Pending |
| 3 | Generated text improves | 🔜 Pending |
| 4 | Reproducible example dataset & documented commands | 🔜 Pending |
| 5 | Scale to 1M–10M parameter model | 🔜 Pending |

### Phase 3 — Inference & CPU Performance

| # | Task | Status |
|---|------|--------|
| 1 | Profile matrix multiplication & Transformer forward | 🔜 Pending |
| 2 | Reusable work buffers | 🔜 Pending |
| 3 | Controlled thread pool | 🔜 Pending |
| 4 | KV caching for autoregressive generation | 🔜 Pending |
| 5 | Benchmark before/after each optimization | 🔜 Pending |

### Phase 4 — BPE Tokenization

| # | Task | Status |
|---|------|--------|
| 1 | Finish/train BPE vocabulary & merge table | ⚠️ Basic implementation exists |
| 2 | Persist tokenizer artifacts with checkpoints | 🔜 Pending |
| 3 | Generation uses exactly the tokenizer used during training | 🔜 Pending |

### Phase 5 — Quantization

| # | Task | Status |
|---|------|--------|
| 1 | Versioned INT8 model format (scales + quantized weights) | ⚠️ Basic quantization exists |
| 2 | Quantized linear/matmul inference | 🔜 Pending |
| 3 | Measure output deviation & memory reduction | 🔜 Pending |
| 4 | INT4 (only after INT8 is correct & tested) | 🔜 Pending |

### Phase 6 — Documentation & Release

| # | Task | Status |
|---|------|--------|
| 1 | README: full quick-start (train → inspect → evaluate → generate) | ⚠️ Basic README exists |
| 2 | Checkpoint format documentation | 🔜 Pending |
| 3 | Supported model sizes & hardware expectations | 🔜 Pending |
| 4 | Release build command → `target/jallm.jar` | ⚠️ Basic packaging exists |

---

## Current Implementation Status

### What Has Been Implemented

#### Core Tensor Engine
- `Tensor.java` — Flat `float[]` storage with row-major indexing
- `TensorShape.java` — Shape validation, flat index calculation, broadcasting
- `Tensor1D.java`, `Tensor2D.java`, `Tensor3D.java` — Specialized tensor views
- Static factories: `zeros()`, `ones()`, `rand()`, `randn()`, `scalar()`
- Element-wise operations: add, sub, mul, div, neg
- Broadcasting support
- Reshape with `-1` inference
- `sumToShape()` for broadcast reductions

#### Mathematical Operations
- `MatrixMath.java` — Matrix multiplication, transpose, batch matmul
- `VectorMath.java` — Vector operations
- `MathUtils.java` — Numerical utilities (stable softmax, etc.)
- `RandomInitializer.java` — Xavier, He, constant, normal initializers

#### Autograd System
- `Variable.java` — Differentiable tensor wrapper
- `ComputationalGraph.java` — Reverse-mode autograd graph
- `VariableOps.java` — Autograd-aware operations (matmul, add, gelu, silu, softmax, etc.)
- `Operation.java` — Operation interface for custom backward passes
- Autograd ops: AddOp, SubOp, MulOp, DivOp, NegOp, MatMulOp, GeluOp, SiLUOp, SoftmaxOp

#### Neural Network Primitives
- `LayerNorm.java` — Layer normalization with stable computation
- `Softmax.java` — Numerically stable softmax
- `GELU.java`, `SiLU.java` — Activation functions
- `TokenEmbedding.java` — Token + position embeddings
- `CausalMask.java` — Causal attention masking
- `ScaledDotProductAttention.java` — Core attention computation

#### Transformer Components
- `MultiHeadAttention.java` — Multi-head attention with autograd & inference paths
- `FeedForward.java` — Position-wise FFN with GELU
- `TransformerBlock.java` — Pre-norm transformer block
- `Transformer.java` — Stack of transformer blocks

#### Model
- `MiniGPT.java` — Decoder-only Transformer (embedding → transformer → LM head)
- `LanguageModel.java` — Model interface

#### Training
- `Trainer.java` — Training loop with autograd & inference modes
- `AdamW.java` — AdamW optimizer with decoupled weight decay
- `Optimizer.java` — Optimizer interface
- `CrossEntropyLoss.java` — Cross-entropy with stable log-softmax + NLL
- `TrainingState.java` — Training metadata (epoch, step, loss, perplexity)
- `Batch.java` — Batch data container

#### Serialization
- `ModelWriter.java` — Write model + training state (gzip-compressed binary)
- `ModelReader.java` — Read model + training state with version checking
- Checkpoint format: magic number (0x4A4C4C4D), version, config, state, parameters

#### Tokenizers
- `CharacterTokenizer.java` — Character-level tokenization
- `WordTokenizer.java` — Word-level tokenization
- `BPETokenizer.java` — Byte Pair Encoding (basic training & encoding)
- `Tokenizer.java` — Tokenizer interface

#### CLI
- `CliDispatcher.java` — Commands: train, generate, evaluate, inspect, benchmark, quantize, selftest, help
- `Main.java` — Entry point with banner, hardware info, tensor self-check

#### Generation
- `GreedySampler.java` — Greedy decoding
- `TemperatureSampler.java` — Temperature sampling
- `TopKSampler.java` — Top-k sampling
- `TopPSampler.java` — Top-p (nucleus) sampling
- `Sampler.java` — Sampler interface

#### Benchmarks
- `MatrixBenchmark.java` — Matrix multiplication benchmarks
- `InferenceBenchmark.java` — Inference benchmarks

#### Configurations
- `ModelConfig.java` — Model architecture configuration
- `TrainingConfig.java` — Training hyperparameters
- `RuntimeConfig.java` — Runtime/inference configuration

#### Quantization
- `Int8Quantizer.java` — INT8 quantization
- `Int4Quantizer.java` — INT4 quantization
- `Quantizer.java`, `QuantizationType.java`, `QuantizedTensor.java`

#### Tests (65 tests, 60 passing, 5 failing)
- `TensorTest` — 4/4 passing
- `MatrixMathTest` — passing
- `AutogradTest` — passing (except testZeroGrad NPE)
- `CrossEntropyLossTest` — passing
- `AttentionTest` — passing
- `LayerNormTest` — passing
- `AdamWTest` — 1 passing, 3 failing
- `TokenizerTest` — 6/7 passing
- `TransformerTest` — 7/7 passing
- `MiniGPTTest` — passing
- `SoftmaxTest` — passing

---

## Pending Work

### Critical Fixes (Test Failures)

| Test | Issue | Root Cause |
|------|-------|------------|
| `AdamWTest.testSimpleOptimization` | Expected loss 5.0, got 12.81 | Test expectation may be wrong for 1 step at lr=0.1, or optimizer needs gradient clipping |
| `AdamWTest.testMultipleParameters` | Expected loss 3.0, got 13.98 | Same issue — test expectations need adjustment |
| `AdamWTest.testZeroGrad` | NPE on `gradient()` returning null | `zeroGrad()` doesn't null out gradients, only clears via `p.zeroGrad()` which may not exist on Variable |
| `AutogradTest.testZeroGrad` | Same NPE issue | Same root cause |
| `TokenizerTest.testTokenizerVocabSizes` | Vocab size assertion fails | BPETokenizer vocab doesn't match expected size after training |

### Phase 1 Completion

- [ ] Fix all test failures
- [ ] Add numerical gradient-check tests for autograd, LayerNorm, attention, cross-entropy
- [ ] Add end-to-end test: tiny corpus → training → lower loss → save/load → generation
- [ ] Make tokenizer vocabulary persistent (save/load alongside checkpoint)
- [ ] Add deterministic seed handling throughout training pipeline
- [ ] Implement checkpoint resume (restore optimizer state, not just parameters)
- [ ] Make `train` CLI support configurable model/training options
- [ ] Add validation errors for short datasets, invalid settings, incompatible checkpoints

### Phase 2 — Model Learning Verification

- [ ] Create reproducible tiny corpus (character-level)
- [ ] Train small model, confirm loss decreases
- [ ] Verify generated text improves over training steps
- [ ] Document commands with example dataset
- [ ] Scale to 1M–10M parameter model

### Phase 3 — Inference & CPU Performance

- [ ] Profile matrix multiplication and Transformer forward passes
- [ ] Add reusable work buffers to reduce allocations
- [ ] Implement controlled thread pool (not unrestricted parallel streams)
- [ ] Implement KV caching for autoregressive generation
- [ ] Benchmark before/after each optimization

### Phase 4 — BPE Tokenization

- [ ] Finish BPE vocabulary and merge table training
- [ ] Persist tokenizer artifacts with checkpoints
- [ ] Ensure generation uses exactly the tokenizer used during training

### Phase 5 — Quantization

- [ ] Define versioned INT8 model format with scales and quantized weight payloads
- [ ] Add quantized linear/matmul inference kernel
- [ ] Measure output deviation and memory reduction
- [ ] Implement INT4 only after INT8 is correct and tested

### Phase 6 — Documentation & Release

- [ ] Update README with full quick-start guide
- [ ] Document checkpoint format, supported model sizes, hardware expectations, limitations
- [ ] Add release build command producing `target/jallm.jar`

---

## Test Strategy

### Test Categories

1. **Unit Tests** — Individual components (Tensor, MatrixMath, LayerNorm, etc.)
2. **Integration Tests** — Multi-component flows (Transformer, MiniGPT forward/backward)
3. **Numerical Tests** — Gradient checks, reference comparison
4. **End-to-End Tests** — Full training → save → load → generate pipeline
5. **Error Tests** — Invalid inputs, edge cases, boundary conditions

### Current Test Results

```
Tests run: 65, Failures: 3, Errors: 2, Skipped: 0
```

**Passing**: Tensor, MatrixMath, Autograd (most), CrossEntropyLoss, Attention, LayerNorm, Transformer, MiniGPT, Softmax

**Failing**: AdamW (3), Autograd zeroGrad (1), Tokenizer vocab (1)

---

## Performance Roadmap

### Optimization Order

```
Correctness → Flat memory → float → stride caching → loop optimization
→ buffer reuse → blocking/tiling → multithreading → SIMD/vector API → quantization
```

### Key Performance Targets

| Operation | Current | Target |
|-----------|---------|--------|
| Matrix multiply (1024×1024) | Naive triple loop | Blocked + parallel |
| Transformer forward (1 layer) | Unoptimized | Buffer reuse + parallel |
| Autoregressive generation | Recompute all layers | KV caching |
| Memory allocation per step | New arrays | Reusable workspaces |

---

## CLI Command Reference

```bash
# Run self-tests
java -jar target/jallm.jar selftest

# Train a model
java -jar target/jallm.jar train <data-file> [output-dir]

# Generate text
java -jar target/jallm.jar generate <model-file> <prompt> [max-tokens] [temperature] [top-k] [top-p] [sampler]

# Evaluate
java -jar target/jallm.jar evaluate <model-file> <data-file> [batch-size]

# Inspect model
java -jar target/jallm.jar inspect <model-file>

# Benchmark
java -jar target/jallm.jar benchmark <matrix|inference|all>

# Quantize
java -jar target/jallm.jar quantize <model-file> <output-file> <int8|int4>
```

---

## Checkpoint Format

```
┌─────────────────────────────────────┐
│ Magic Number: 0x4A4C4C4D ("JLLM")  │  4 bytes
│ Version: 2                          │  4 bytes
├─────────────────────────────────────┤
│ Model Config:                       │
│   - Model type string               │
│   - vocabSize, embedDim             │
│   - maxSeqLen, numLayers            │
│   - numHeads, ffDim                 │
│   - dropout                         │
├─────────────────────────────────────┤
│ Training State:                     │
│   - epoch, step, globalStep         │
│   - learningRate, trainLoss         │
│   - valLoss, perplexity             │
│   - timestamp                       │
│   - modelConfigJson                 │
├─────────────────────────────────────┤
│ Parameters:                         │
│   - Parameter count                 │  4 bytes
│   - For each parameter:             │
│     - Name (UTF string)             │
│     - Rank (4 bytes)                │
│     - Shape (rank × 4 bytes)        │
│     - Element count (4 bytes)       │
│     - Float data (count × 4 bytes)  │
└─────────────────────────────────────┘
```

Compressed with GZIP.

---

## Hardware & Memory Budget

| RAM | Conservative Model | Comfortable Model | Experimental |
|-----|--------------------|--------------------|--------------|
| 8 GB | ~1M params | ~3M params | Not recommended |
| 16 GB | ~3M params | ~10M params | ~15M params |
| 32 GB | ~10M params | ~30M params | ~50M params |

FP32 memory: `params × 4 bytes`. AdamW adds `params × 8 bytes` (m + v). Total training memory ≈ `params × 12 bytes`.

---

## Release Criteria

A release build must produce `target/jallm.jar` and satisfy:

1. All tests pass (0 failures, 0 errors)
2. `selftest` command completes successfully
3. End-to-end training on tiny corpus produces decreasing loss
4. Saved checkpoint can be reloaded and used for generation
5. Evaluation reports valid loss and perplexity
6. Quantization produces measurable output deviation (documented)
7. CLI commands provide useful error messages for invalid inputs
8. README documents quick-start, checkpoint format, and limitations
