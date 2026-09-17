# JALLM — Java LLM From Scratch

> **BETA** — A decoder-only Transformer language model implemented and trained from random weights in pure Java. No external AI APIs, no PyTorch, no TensorFlow, no llama.cpp. Everything is built from scratch.

![Status](https://img.shields.io/badge/status-beta-yellow)
![Java](https://img.shields.io/badge/Java-21-blue)
![Tests](https://img.shields.io/badge/tests-67%20passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

---

## Table of Contents

- [What Is JALLM?](#what-is-jallm)
- [Quick Start](#quick-start)
- [System Requirements](#system-requirements)
- [Installation](#installation)
  - [Linux / macOS](#linux--macos)
  - [Windows](#windows)
  - [Cross-Platform (Python)](#cross-platform-python)
- [Verify Installation](#verify-installation)
- [Usage](#usage)
  - [Train](#train)
  - [Generate](#generate)
  - [Evaluate](#evaluate)
  - [Inspect](#inspect)
  - [Benchmark](#benchmark)
  - [Quantize](#quantize)
- [Project Structure](#project-structure)
- [Checkpoint Format](#checkpoint-format)
- [Hardware Expectations](#hardware-expectations)
- [Limitations](#limitations)
- [Contributing](#contributing)
- [License](#license)

---

## What Is JALLM?

**JALLM** (Java LLM From Scratch) is a decoder-only Transformer language model built entirely in Java. It demonstrates:

- Contiguous `float[]` tensor storage with row-major indexing
- Reverse-mode autograd (automatic differentiation)
- Causal multi-head self-attention
- Transformer blocks with LayerNorm
- AdamW optimizer
- Character, word, and BPE tokenization
- Checkpoint save/load with training state
- Text generation with greedy, temperature, top-k, and top-p sampling
- INT8/INT4 quantization (experimental)

**This is a learning/reference implementation.** It is not optimized for production use.

---

## Quick Start

### Prerequisites

- **Java Development Kit (JDK) 21** or later
- **Maven 3.9+** (for building from source)

### Build and Run

```bash
# Clone the repository
git clone https://github.com/your-org/jallm.git
cd jallm

# Build
mvn clean package

# Run self-test
java -jar target/jallm.jar selftest

# Train a model
echo "hello world hello java" > data/train.txt
java -jar target/jallm.jar train data/train.txt models/checkpoints

# Generate text
java -jar target/jallm.jar generate models/checkpoints/model.bin "hello" 50

# Evaluate
java -jar target/jallm.jar evaluate models/checkpoints/model.bin data/train.txt

# Inspect model
java -jar target/jallm.jar inspect models/checkpoints/model.bin
```

---

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **OS** | Windows 10+, macOS 11+, Linux kernel 5.x+ | Latest stable |
| **Java** | JDK 21 | JDK 21+ |
| **RAM** | 2 GB | 8 GB+ |
| **Disk** | 200 MB | 1 GB+ |
| **CPU** | Any x86_64/ARM64 | 4+ cores |
| **Maven** | 3.9+ | 3.9+ |

### Java Installation

#### Linux
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install openjdk-21-jdk maven

# Fedora/RHEL
sudo dnf install java-21-openjdk maven

# Arch
sudo pacman -S jdk21-openjdk maven
```

#### macOS
```bash
# Install Homebrew if not installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install JDK 21 and Maven
brew install --cask temurin@21
brew install maven
```

#### Windows
```powershell
# Using Chocolatey
choco install jdk21 maven

# Using Scoop
scoop install java maven
```

---

## Installation

### Linux / macOS

#### Option 1: Install Script

```bash
# Download and run the installer
curl -fsSL https://raw.githubusercontent.com/your-org/jallm/main/install.sh | bash

# Or clone and run manually
git clone https://github.com/your-org/jallm.git
cd jallm
chmod +x install.sh
./install.sh
```

The `install.sh` script will:
1. Check for JDK 21 and Maven
2. Install them if missing (via system package manager)
3. Build the project with Maven
4. Run the self-test
5. Create necessary directories (`models/`, `data/`, `logs/`)

#### Option 2: Manual

```bash
# Prerequisites
sudo apt install openjdk-21-jdk maven  # Debian/Ubuntu
# or
brew install --cask temurin@21 maven   # macOS

# Build
git clone https://github.com/your-org/jallm.git
cd jallm
mvn clean package

# Verify
java -jar target/jallm.jar selftest
```

### Windows

#### Option 1: Batch Script

```cmd
@echo off
echo ========================================
echo   JALLM - Java LLM From Scratch (Beta)
echo ========================================
echo.

:: Check for Java
java -version 2>&1 | findstr "version" >nul
if %errorlevel% neq 0 (
    echo [ERROR] Java not found. Please install JDK 21+.
    echo Download: https://adoptium.net/
    pause
    exit /b 1
)

:: Check for Maven
mvn -version 2>&1 | findstr "Maven" >nul
if %errorlevel% neq 0 (
    echo [ERROR] Maven not found. Please install Maven 3.9+.
    echo Download: https://maven.apache.org/download.cgi
    pause
    exit /b 1
)

echo [INFO] Building JALLM...
mvn clean package

echo [INFO] Running self-test...
java -jar target/jallm.jar selftest

echo [INFO] Installation complete!
echo.
echo Usage:
echo   java -jar target/jallm.jar train ^<data-file^> [output-dir]
echo   java -jar target/jallm.jar generate ^<model^> ^<prompt^> [max-tokens]
echo   java -jar target/jallm.jar help
pause
```

#### Option 2: PowerShell Script

```powershell
# install.ps1
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  JALLM - Java LLM From Scratch (Beta)" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Check Java
$java = Get-Command java -ErrorAction SilentlyContinue
if (-not $java) {
    Write-Host "[ERROR] Java not found. Install JDK 21+ from https://adoptium.net/" -ForegroundColor Red
    exit 1
}

# Check Maven
$mvn = Get-Command mvn -ErrorAction SilentlyContinue
if (-not $mvn) {
    Write-Host "[ERROR] Maven not found. Install from https://maven.apache.org/download.cgi" -ForegroundColor Red
    exit 1
}

Write-Host "[INFO] Building JALLM..." -ForegroundColor Green
mvn clean package

Write-Host "[INFO] Running self-test..." -ForegroundColor Green
java -jar target/jallm.jar selftest

Write-Host "[INFO] Installation complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Usage:" -ForegroundColor Yellow
Write-Host "  java -jar target/jallm.jar train <data-file> [output-dir]"
Write-Host "  java -jar target/jallm.jar generate <model> <prompt> [max-tokens]"
```

#### Option 3: Manual (GUI)

1. Download the latest release JAR from the releases page
2. Double-click `jallm.jar` (requires Java to be associated with JAR files)
3. Or run from Command Prompt: `java -jar jallm.jar help`

### Cross-Platform (Python)

The `install.py` script works on **Linux, macOS, and Windows**:

```bash
# Run the installer
python3 install.py

# Options
python3 install.py --no-build     # Skip Maven build
python3 install.py --java-path /path/to/java  # Specify Java path
python3 install.py --verbose      # Verbose output
python3 install.py --help         # Show help
```

The script:
1. Detects your operating system
2. Checks/installs prerequisites (Java, Maven)
3. Builds the project
4. Runs self-tests
5. Creates directory structure
6. Provides usage instructions

---

## Verify Installation

After installation, run:

```bash
java -jar target/jallm.jar selftest
```

Expected output:

```
=========================================
        JALLM - LLM FROM SCRATCH
=========================================

  Version:       0.1-BETA
  Backend:       Java CPU
  Precision:     FP32
  External AI:   NONE
  Status:        INITIALIZED
=========================================

  Hardware:
    OS:           Linux 7.2.6
    Processors:   8 cores
    Arch:         amd64
    Max Heap:     7980 MB
=========================================

  Tensor engine self-check:
    zeros(2,3) -> shape=[2, 3] size=6 elements=6
    All values == 0.0: true
=========================================

Tensor engine self-test: PASSED
```

If you see `Tensor engine self-test: PASSED`, installation was successful.

---

## Usage

### Train

Train a model from a text corpus:

```bash
# Basic training
java -jar target/jallm.jar train data/train.txt

# With custom output directory
java -jar target/jallm.jar train data/train.txt models/my-model

# The training uses a character-level tokenizer by default
# Model is saved as a gzip-compressed binary checkpoint
```

**Training output:**
```
Loaded training data: 1024 characters
Tokenized: 1030 tokens, vocab size: 98
Created 52 training batches
Training completed. Final loss: 0.487
Model saved to models/checkpoints/model.bin
```

### Generate

Generate text from a trained model:

```bash
# Basic generation (temperature sampling)
java -jar target/jallm.jar generate models/checkpoints/model.bin "hello" 50

# Greedy decoding
java -jar target/jallm.jar generate models/checkpoints/model.bin "hello" 50 1.0 0 1.0 greedy

# Temperature sampling
java -jar target/jallm.jar generate models/checkpoints/model.bin "hello" 50 0.7 0 1.0 temperature

# Top-k sampling
java -jar target/jallm.jar generate models/checkpoints/model.bin "hello" 50 1.0 5 1.0 topk

# Top-p (nucleus) sampling
java -jar target/jallm.jar generate models/checkpoints/model.bin "hello" 50 1.0 0 0.9 topp
```

**Arguments:**
| Position | Description | Default |
|----------|-------------|---------|
| 1 | Model file path | (required) |
| 2 | Prompt text | (required) |
| 3 | Max tokens to generate | 100 |
| 4 | Temperature | 1.0 |
| 5 | Top-k (0 = disabled) | 0 |
| 6 | Top-p (0 = disabled) | 1.0 |
| 7 | Sampler type | temperature |

### Evaluate

Evaluate a model on a dataset:

```bash
java -jar target/jallm.jar evaluate models/checkpoints/model.bin data/train.txt

# With custom batch size
java -jar target/jallm.jar evaluate models/checkpoints/model.bin data/train.txt 8
```

**Output:**
```
Loaded model: 16d, 1L, 1H
Created 52 evaluation batches
Evaluation completed on 52 batches. Loss: 0.487516, perplexity: 1.6284
```

### Inspect

Inspect a model checkpoint:

```bash
java -jar target/jallm.jar inspect models/checkpoints/model.bin
```

**Output:**
```
=== Model Inspection ===
Architecture: MiniGPT (Decoder-only Transformer)
Embedding dim: 16
Layers: 1
Heads: 1
Head dim: 16
FFN dim: 64
Vocab size: 98
Max seq len: 8
Dropout: 0.1
Total parameters: 5106 (0.01M)
Memory (FP32): 0.02 MB
Memory (INT8): 0.01 MB
Memory (INT4): 0.00 MB
```

### Benchmark

Run performance benchmarks:

```bash
java -jar target/jallm.jar benchmark matrix      # Matrix multiplication
java -jar target/jallm.jar benchmark inference    # Inference benchmark
java -jar target/jallm.jar benchmark all          # Both
```

### Quantize

Quantize a model (experimental):

```bash
java -jar target/jallm.jar quantize models/checkpoints/model.bin models/quantized/int8.bin int8
java -jar target/jallm.jar quantize models/checkpoints/model.bin models/quantized/int4.bin int4
```

---

## Project Structure

```
jallm/
├── pom.xml                    # Maven configuration
├── README.md                  # This file
├── install.sh                 # Linux/macOS installer
├── install.py                 # Cross-platform installer
├── install.bat                # Windows installer
├── docs/
│   ├── project_plan.md        # Full project plan
│   └── plan_implementation.md # Implementation status
├── data/
│   ├── raw/                   # Raw training data
│   │   └── tiny.txt           # Example tiny corpus
│   ├── processed/             # Processed data
│   └── tokenizer/             # Tokenizer artifacts
├── models/
│   ├── checkpoints/           # Saved checkpoints
│   └── final/                 # Final models
├── src/
│   ├── main/java/com/jalllm/geforce/
│   │   ├── Main.java          # Entry point
│   │   ├── CliDispatcher.java # CLI commands
│   │   ├── tensor/            # Tensor engine
│   │   ├── math/              # Matrix/vector math
│   │   ├── autograd/          # Automatic differentiation
│   │   ├── attention/         # Attention mechanisms
│   │   ├── transformer/       # Transformer blocks
│   │   ├── model/             # Model architectures
│   │   ├── training/          # Training loop
│   │   ├── optimizer/         # AdamW optimizer
│   │   ├── loss/              # Loss functions
│   │   ├── tokenizer/         # Tokenizers
│   │   ├── embedding/         # Token + position embeddings
│   │   ├── normalization/     # LayerNorm
│   │   ├── generation/        # Text generation/sampling
│   │   ├── serialization/     # Checkpoint I/O
│   │   ├── config/            # Model/training/runtime config
│   │   ├── quantization/      # INT8/INT4 quantization
│   │   └── benchmark/         # Performance benchmarks
│   └── test/java/com/jalllm/geforce/
│       └── ...                # Unit and integration tests
└── target/
    └── jallm.jar              # Built JAR
```

---

## Checkpoint Format

Checkpoints are gzip-compressed binary files with the following structure:

```
┌─────────────────────────────────────┐
│ Magic: 0x4A4C4C4D ("JLLM")       │  4 bytes
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

---

## Hardware Expectations

| RAM | Conservative Model | Comfortable Model | Experimental |
|-----|--------------------|--------------------|--------------|
| 2 GB | ~100K params | Not recommended | — |
| 4 GB | ~100K params | ~500K params | Not recommended |
| 8 GB | ~500K params | ~2M params | ~5M params |
| 16 GB | ~2M params | ~10M params | ~20M params |
| 32 GB | ~10M params | ~30M params | ~50M params |

**Memory estimate:** FP32 weights = `params × 4 bytes`. AdamW optimizer adds `params × 8 bytes` (m + v). Total training memory ≈ `params × 12 bytes`.

---

## Limitations

This is a **Beta** learning implementation. Known limitations:

1. **CPU only** — No GPU acceleration (CUDA/Metal/Vulkan)
2. **FP32 only** — No FP16/BF16 inference (except experimental quantization)
3. **No KV caching** — Autoregressive generation recomputes all layers each step
4. **No learning rate scheduling** — Constant learning rate
5. **No gradient clipping** — Clipping is configured but not enforced
6. **Single-threaded training** — No parallel execution
7. **Character-level tokenizer** — BPE is basic; word tokenizer is functional
8. **No mixed precision** — Training is FP32 throughout
9. **Quantization is demonstration only** — INT8/INT4 quantized inference kernels not implemented
10. **No streaming data** — Entire corpus must fit in memory

These limitations are intentional for a learning implementation and will be addressed in future releases.

---

## Contributing

This is an open Beta project. Contributions are welcome!

### How to Contribute

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes
4. Run tests: `mvn test`
5. Commit: `git commit -m 'Add amazing feature'`
6. Push: `git push origin feature/amazing-feature`
7. Open a Pull Request

### Development Setup

```bash
# Clone
git clone https://github.com/your-org/jallm.git
cd jallm

# Build
mvn clean compile

# Run tests
mvn test

# Run self-test
java -jar target/jallm.jar selftest

# Run with debug
java -jar target/jallm.jar help
```

### Code Style

- Java 21 features are used (pattern matching, records, etc.)
- Follow existing code style (comments, naming conventions)
- All public methods should have Javadoc
- All new features must include tests

---

## License

MIT License

Copyright (c) 2026 JALLM Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

