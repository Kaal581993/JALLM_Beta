#!/usr/bin/env bash
# ============================================================
# File Name    : install.sh
# Location     : install.sh
# Author       : Viral Prajapati
# Date         : 15-09-2026
#
# Description:
#   Linux / macOS installer for JALLM (Java LLM From Scratch).
#   This script checks for prerequisites, installs them if
#   missing, builds the project, and runs self-tests.
#
# Usage:
#   chmod +x install.sh
#   ./install.sh
#   curl -fsSL https://raw.githubusercontent.com/your-org/jallm/main/install.sh | bash
# ============================================================

set -euo pipefail

# ── Colors ──────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# ── Helpers ─────────────────────────────────────────────────
info()  { echo -e "${BLUE}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; }
banner(){ echo -e "${CYAN}$*${NC}"; }

# ── Banner ──────────────────────────────────────────────────
banner "========================================"
banner "  JALLM - Java LLM From Scratch (Beta)"
banner "========================================"
echo ""

# ── Detect OS ───────────────────────────────────────────────
OS="unknown"
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
else
    warn "Unrecognized OS: $OSTYPE"
    OS="unknown"
fi
info "Detected OS: $OS"

# ── Check / Install Java (JDK 21+) ─────────────────────────
check_java() {
    if command -v java &>/dev/null; then
        JAVA_VERSION=$(java -version 2>&1 | head -1 | cut -d'"' -f2 | sed '/^1\./s///' | cut -d'.' -f1)
        if [[ "$JAVA_VERSION" -ge 21 ]]; then
            ok "Java found: $(java -version 2>&1 | head -1)"
            return 0
        else
            warn "Java version $JAVA_VERSION is too old. Need JDK 21+."
        fi
    else
        warn "Java not found."
    fi
    return 1
}

install_java() {
    info "Installing JDK 21..."
    if [[ "$OS" == "linux" ]]; then
        if command -v apt-get &>/dev/null; then
            sudo apt-get update -qq
            sudo apt-get install -y -qq openjdk-21-jdk maven
        elif command -v dnf &>/dev/null; then
            sudo dnf install -y java-21-openjdk maven
        elif command -v pacman &>/dev/null; then
            sudo pacman -Sy --noconfirm jdk21-openjdk maven
        elif command -v yum &>/dev/null; then
            sudo yum install -y java-21-openjdk maven
        else
            error "No supported package manager found. Please install JDK 21+ and Maven manually."
            exit 1
        fi
    elif [[ "$OS" == "macos" ]]; then
        if command -v brew &>/dev/null; then
            brew install --cask temurin@21
            brew install maven
        else
            error "Homebrew not found. Please install JDK 21+ and Maven manually."
            exit 1
        fi
    else
        error "Cannot auto-install Java on this OS. Please install JDK 21+ and Maven manually."
        exit 1
    fi
}

if ! check_java; then
    install_java
fi

# Verify Java after potential install
if ! command -v java &>/dev/null; then
    error "Java installation failed or is not in PATH."
    exit 1
fi
ok "Java: $(java -version 2>&1 | head -1)"

# ── Check / Install Maven ───────────────────────────────────
if command -v mvn &>/dev/null; then
    MVN_VERSION=$(mvn -version 2>&1 | head -1)
    ok "Maven found: $MVN_VERSION"
else
    warn "Maven not found. Installing..."
    if [[ "$OS" == "linux" ]]; then
        if command -v apt-get &>/dev/null; then
            sudo apt-get install -y -qq maven
        elif command -v dnf &>/dev/null; then
            sudo dnf install -y maven
        elif command -v pacman &>/dev/null; then
            sudo pacman -Sy --noconfirm maven
        else
            error "Cannot auto-install Maven. Please install Maven 3.9+ manually."
            exit 1
        fi
    elif [[ "$OS" == "macos" ]]; then
        if command -v brew &>/dev/null; then
            brew install maven
        else
            error "Homebrew not found. Please install Maven manually."
            exit 1
        fi
    else
        error "Cannot auto-install Maven on this OS."
        exit 1
    fi
fi

if ! command -v mvn &>/dev/null; then
    error "Maven installation failed."
    exit 1
fi
ok "Maven: $(mvn -version 2>&1 | head -1)"

# ── Determine Project Directory ─────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"

if [[ ! -f "$PROJECT_DIR/pom.xml" ]]; then
    error "pom.xml not found in $PROJECT_DIR"
    error "Please run this script from the project root directory."
    exit 1
fi
ok "Project directory: $PROJECT_DIR"
cd "$PROJECT_DIR"

# ── Create Required Directories ─────────────────────────────
info "Creating directories..."
mkdir -p models/checkpoints models/final data/raw data/processed data/tokenizer logs
ok "Directories created."

# ── Build Project ───────────────────────────────────────────
banner ""
banner "Building JALLM..."
banner ""

info "Running: mvn clean package"
if mvn clean package -q; then
    ok "Build successful."
else
    error "Build failed. Check the output above for errors."
    exit 1
fi

# ── Verify JAR ──────────────────────────────────────────────
if [[ -f "$PROJECT_DIR/target/jallm.jar" ]]; then
    ok "JAR created: target/jallm.jar"
else
    warn "target/jallm.jar not found. Checking for any JAR..."
    JAR=$(find target -name "*.jar" | head -1)
    if [[ -n "$JAR" ]]; then
        ok "Found JAR: $JAR"
    else
        error "No JAR file found after build."
        exit 1
    fi
fi

# ── Run Self-Test ───────────────────────────────────────────
banner ""
banner "Running self-test..."
banner ""

if java -jar target/jallm.jar selftest; then
    echo ""
    ok "Self-test PASSED!"
else
    error "Self-test FAILED."
    exit 1
fi

# ── Summary ─────────────────────────────────────────────────
echo ""
banner "========================================"
banner "  Installation Complete!"
banner "========================================"
echo ""
echo -e "${GREEN}Project:${NC}  JALLM (Java LLM From Scratch)"
echo -e "${GREEN}Version:${NC}  0.1-BETA"
echo -e "${GREEN}Java:${NC}    $(java -version 2>&1 | head -1)"
echo -e "${GREEN}OS:${NC}      $OS"
echo ""
echo "Quick start:"
echo "  java -jar target/jallm.jar help"
echo "  java -jar target/jallm.jar train data/train.txt"
echo "  java -jar target/jallm.jar generate models/checkpoints/model.bin \"hello\" 50"
echo ""
echo "For more information, see README.md"
echo ""
