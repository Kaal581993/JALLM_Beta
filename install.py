#!/usr/bin/env python3
"""
File Name    : install.py
Location     : install.py
Author       : Viral Prajapati
Date         : 15-09-2026

Description:
    Cross-platform installer for JALLM (Java LLM From Scratch).
    Works on Linux, macOS, and Windows.

    This script checks for prerequisites (Java JDK 21+, Maven 3.9+),
    installs them if missing, builds the project with Maven,
    runs self-tests, and creates necessary directories.

Usage:
    python3 install.py
    python3 install.py --no-build
    python3 install.py --java-path /path/to/java
    python3 install.py --verbose
    python3 install.py --help
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

# Colors for terminal output
class C:
    RED = "\033[0;31m"
    GREEN = "\033[0;32m"
    YELLOW = "\033[1;33m"
    BLUE = "\033[0;34m"
    CYAN = "\033[0;36m"
    NC = "\033[0m"

    @staticmethod
    def info(msg):
        return f"{C.BLUE}[INFO]{C.NC}  {msg}"

    @staticmethod
    def ok(msg):
        return f"{C.GREEN}[OK]{C.NC}    {msg}"

    @staticmethod
    def warn(msg):
        return f"{C.YELLOW}[WARN]{C.NC}  {msg}"

    @staticmethod
    def error(msg):
        return f"{C.RED}[ERROR]{C.NC} {msg}"

    @staticmethod
    def banner(msg):
        return f"{C.CYAN}{msg}{C.NC}"


def print_banner():
    print(C.banner("========================================"))
    print(C.banner("  JALLM - Java LLM From Scratch (Beta)"))
    print(C.banner("========================================"))
    print()


def run_command(cmd, check=True, capture=False, verbose=False):
    """Run a system command and return the result."""
    if verbose:
        print(C.info(f"Running: {' '.join(cmd) if isinstance(cmd, list) else cmd}"))
    try:
        result = subprocess.run(
            cmd,
            check=check,
            capture_output=capture,
            text=True,
        )
        return result
    except subprocess.CalledProcessError as e:
        if check:
            print(C.error(f"Command failed: {e}"))
            raise
        return e


def command_exists(cmd):
    """Check if a command exists in PATH."""
    return shutil.which(cmd) is not None


def get_java_version():
    """Get the Java version as an integer (major version)."""
    try:
        result = run_command(["java", "-version"], capture=True, check=False)
        output = (result.stderr + result.stdout) if result.stderr else result.stdout
        first_line = output.split("\n")[0]
        if '"' in first_line:
            version_str = first_line.split('"')[1]
        else:
            version_str = first_line.split()[-1]
        if version_str.startswith("1."):
            return int(version_str.split(".")[1])
        return int(version_str.split(".")[0])
    except Exception:
        return None


def detect_os():
    """Detect the operating system."""
    system = platform.system().lower()
    if system == "linux":
        return "linux"
    elif system == "darwin":
        return "macos"
    elif system == "windows":
        return "windows"
    return "unknown"


def install_java(os_name, verbose=False):
    """Attempt to install JDK 21+ on the detected OS."""
    print(C.info("Installing JDK 21+..."))

    if os_name == "linux":
        if command_exists("apt-get"):
            run_command(["sudo", "apt-get", "update", "-qq"], verbose=verbose)
            run_command(["sudo", "apt-get", "install", "-y", "-qq", "openjdk-21-jdk"], verbose=verbose)
        elif command_exists("dnf"):
            run_command(["sudo", "dnf", "install", "-y", "java-21-openjdk"], verbose=verbose)
        elif command_exists("pacman"):
            run_command(["sudo", "pacman", "-Sy", "--noconfirm", "jdk21-openjdk"], verbose=verbose)
        elif command_exists("yum"):
            run_command(["sudo", "yum", "install", "-y", "java-21-openjdk"], verbose=verbose)
        else:
            print(C.error("No supported package manager found. Please install JDK 21+ manually."))
            return False
    elif os_name == "macos":
        if command_exists("brew"):
            run_command(["brew", "install", "--cask", "temurin@21"], verbose=verbose)
        else:
            print(C.error("Homebrew not found. Please install JDK 21+ manually."))
            return False
    elif os_name == "windows":
        print(C.error("Automatic Java installation is not supported on Windows."))
        print("Please download and install JDK 21+ from https://adoptium.net/")
        return False
    else:
        print(C.error("Cannot auto-install Java on this OS."))
        return False

    return True


def install_maven(os_name, verbose=False):
    """Attempt to install Maven on the detected OS."""
    print(C.info("Installing Maven..."))

    if os_name == "linux":
        if command_exists("apt-get"):
            run_command(["sudo", "apt-get", "install", "-y", "-qq", "maven"], verbose=verbose)
        elif command_exists("dnf"):
            run_command(["sudo", "dnf", "install", "-y", "maven"], verbose=verbose)
        elif command_exists("pacman"):
            run_command(["sudo", "pacman", "-Sy", "--noconfirm", "maven"], verbose=verbose)
        else:
            print(C.error("Cannot auto-install Maven. Please install Maven 3.9+ manually."))
            return False
    elif os_name == "macos":
        if command_exists("brew"):
            run_command(["brew", "install", "maven"], verbose=verbose)
        else:
            print(C.error("Homebrew not found. Please install Maven manually."))
            return False
    elif os_name == "windows":
        print(C.error("Automatic Maven installation is not supported on Windows."))
        print("Please download and install Maven from https://maven.apache.org/download.cgi")
        return False

    return True


def create_directories(project_dir):
    """Create required project directories."""
    print(C.info("Creating directories..."))
    dirs = [
        "models/checkpoints",
        "models/final",
        "data/raw",
        "data/processed",
        "data/tokenizer",
        "logs",
    ]
    for d in dirs:
        Path(project_dir, d).mkdir(parents=True, exist_ok=True)
    print(C.ok("Directories created."))


def build_project(project_dir, verbose=False):
    """Build the project with Maven."""
    print()
    print(C.banner("Building JALLM..."))
    print()
    print(C.info("Running: mvn clean package"))

    cmd = ["mvn", "clean", "package"]
    result = run_command(cmd, capture=not verbose, verbose=verbose)

    if result.returncode == 0:
        print(C.ok("Build successful."))
        return True
    else:
        print(C.error("Build failed. Check the output above for errors."))
        return False


def run_self_test(project_dir, verbose=False):
    """Run the self-test."""
    print()
    print(C.banner("Running self-test..."))
    print()

    jar_path = str(Path(project_dir, "target", "jallm.jar"))
    if not os.path.exists(jar_path):
        target_dir = Path(project_dir, "target")
        jars = list(target_dir.glob("*.jar")) if target_dir.exists() else []
        if jars:
            jar_path = str(jars[0])
            print(C.warn(f"Using found JAR: {jar_path}"))
        else:
            print(C.error("No JAR file found. Build may have failed."))
            return False

    cmd = ["java", "-jar", jar_path, "selftest"]
    result = run_command(cmd, capture=not verbose, verbose=verbose)

    if result.returncode == 0:
        print()
        print(C.ok("Self-test PASSED!"))
        return True
    else:
        print(C.error("Self-test FAILED."))
        return False


def main():
    parser = argparse.ArgumentParser(
        description="JALLM (Java LLM From Scratch) - Cross-platform installer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 install.py                    # Full install
  python3 install.py --no-build         # Check prerequisites only
  python3 install.py --verbose          # Verbose output
  python3 install.py --java-path /usr/lib/jvm/java-21  # Custom Java path
        """,
    )
    parser.add_argument("--no-build", action="store_true", help="Skip Maven build step")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--java-path", type=str, help="Path to Java installation")
    parser.add_argument("--skip-java-check", action="store_true", help="Skip Java installation check")
    parser.add_argument("--skip-maven-check", action="store_true", help="Skip Maven installation check")
    args = parser.parse_args()

    print_banner()

    os_name = detect_os()
    print(C.info(f"Detected OS: {os_name}"))
    print(C.info(f"Python: {sys.version.split()[0]}"))
    print()

    project_dir = str(Path(__file__).resolve().parent)
    print(C.info(f"Project directory: {project_dir}"))

    if not os.path.exists(os.path.join(project_dir, "pom.xml")):
        print(C.error("pom.xml not found. Please run this script from the project root."))
        sys.exit(1)

    # Check / Install Java
    if not args.skip_java_check:
        java_version = get_java_version()
        if java_version and java_version >= 21:
            print(C.ok(f"Java found: version {java_version}"))
        else:
            if java_version:
                print(C.warn(f"Java version {java_version} is too old. Need JDK 21+."))
            else:
                print(C.warn("Java not found."))

            if not install_java(os_name, args.verbose):
                print(C.error("Java installation failed. Please install JDK 21+ manually."))
                sys.exit(1)

            java_version = get_java_version()
            if java_version and java_version >= 21:
                print(C.ok(f"Java: version {java_version}"))
            else:
                print(C.error("Java verification failed after installation."))
                sys.exit(1)

    # Check / Install Maven
    if not args.skip_maven_check:
        if command_exists("mvn"):
            mv = run_command(["mvn", "-version"], capture=True).stdout.split("\n")[0]
            print(C.ok(f"Maven found: {mv}"))
        else:
            print(C.warn("Maven not found. Installing..."))
            if not install_maven(os_name, args.verbose):
                print(C.error("Maven installation failed. Please install Maven 3.9+ manually."))
                sys.exit(1)

            if command_exists("mvn"):
                mv = run_command(["mvn", "-version"], capture=True).stdout.split("\n")[0]
                print(C.ok(f"Maven: {mv}"))
            else:
                print(C.error("Maven verification failed."))
                sys.exit(1)

    # Create Directories
    create_directories(project_dir)

    # Build
    if not args.no_build:
        if not build_project(project_dir, args.verbose):
            sys.exit(1)

        # Verify JAR
        jar_path = Path(project_dir, "target", "jallm.jar")
        if jar_path.exists():
            print(C.ok("JAR created: target/jallm.jar"))
        else:
            print(C.warn("target/jallm.jar not found, but build may have succeeded with a different name."))

        # Self-Test
        if not run_self_test(project_dir, args.verbose):
            sys.exit(1)
    else:
        print(C.info("Build skipped (--no-build)."))

    # Summary
    print()
    print(C.banner("========================================"))
    print(C.banner("  Installation Complete!"))
    print(C.banner("========================================"))
    print()
    print(f"{C.GREEN}Project:{C.NC}  JALLM (Java LLM From Scratch)")
    print(f"{C.GREEN}Version:{C.NC}  0.1-BETA")
    print(f"{C.GREEN}OS:{C.NC}      {os_name}")
    print()
    print("Quick start:")
    print("  java -jar target/jallm.jar help")
    print("  java -jar target/jallm.jar train data/train.txt")
    print("  java -jar target/jallm.jar generate models/checkpoints/model.bin \"hello\" 50")
    print()
    print("For more information, see README.md")
    print()


if __name__ == "__main__":
    main()
