@echo off
REM ============================================================
REM File Name    : install.bat
REM Location     : install.bat
REM Author       : Viral Prajapati
REM Date         : 15-09-2026
REM
REM Description:
REM   Windows installer for JALLM (Java LLM From Scratch).
REM   This script checks for prerequisites (Java JDK 21+,
REM   Maven 3.9+), builds the project with Maven,
REM   runs self-tests, and creates necessary directories.
REM
REM Usage:
REM   install.bat
REM   install.bat --no-build
REM   install.bat --verbose
REM ============================================================

setlocal enabledelayedexpansion

REM ── Colors ──────────────────────────────────────────
set RED=[31m
set GREEN=[32m
set YELLOW=[33m
set BLUE=[34m
set CYAN=[36m
set NC=[0m

set "RED=%RED%"
set "GREEN=%GREEN%"
set "YELLOW=%YELLOW%"
set "BLUE=%BLUE%"
set "CYAN=%CYAN%"
set "NC=%NC%"

REM ── Helpers ─────────────────────────────────────────
:info
  echo %BLUE%[INFO]%NC%  %~1
  goto :eof

:ok
  echo %GREEN%[OK]%NC%    %~1
  goto :eof

:warn
  echo %YELLOW%[WARN]%NC%  %~1
  goto :eof

:error
  echo %RED%[ERROR]%NC% %~1
  goto :eof

:banner
  echo %CYAN%%~1%NC%
  goto :eof

REM ── Banner ──────────────────────────────────────────
call :banner "========================================"
call :banner "  JALLM - Java LLM From Scratch (Beta)"
call :banner "========================================"
echo.

REM ── Check Java ──────────────────────────────────────
set JAVA_FOUND=0
where java >nul 2>&1
if %errorlevel% equ 0 (
  for /f "tokens=3" %%a in ('java -version 2^>^&1 ^| findstr /i "version"') do (
    set JAVA_VER=%%a
  )
  REM Extract major version number from quoted string like "21.0.x"
  set JAVA_VER=!JAVA_VER:"=!
  for /f "delims=. tokens=1" %%b in ("!JAVA_VER!") do set JAVA_MAJOR=%%b
  if !JAVA_MAJOR! geq 21 (
    call :ok "Java found: !JAVA_VER!"
    set JAVA_FOUND=1
  ) else (
    call :warn "Java version !JAVA_MAJOR! is too old. Need JDK 21+."
  )
)

if !JAVA_FOUND! equ 0 (
  call :error "Java not found or version too old."
  echo.
  echo Please install JDK 21+ from: https://adoptium.net/
  echo.
  echo After installing Java, re-run this script.
  pause
  exit /b 1
)

REM ── Check Maven ─────────────────────────────────────
set MVN_FOUND=0
where mvn >nul 2>&1
if %errorlevel% equ 0 (
  for /f "tokens=1" %%a in ('mvn -version 2^>^&1 ^| findstr /i "Maven"') do (
    call :ok "Maven found: %%a"
    set MVN_FOUND=1
  )
)

if !MVN_FOUND! equ 0 (
  call :warn "Maven not found."
  echo.
  echo Please install Maven 3.9+ from: https://maven.apache.org/download.cgi
  echo.
  echo After installing Maven, re-run this script.
  pause
  exit /b 1
)

REM ── Determine Project Directory ─────────────────────
set "PROJECT_DIR=%~dp0"
REM Remove trailing backslash
if "%PROJECT_DIR:~-1%"=="\" set "PROJECT_DIR=%PROJECT_DIR:~0,-1%"

if not exist "%PROJECT_DIR%\pom.xml" (
  call :error "pom.xml not found in %PROJECT_DIR%"
  pause
  exit /b 1
)
call :ok "Project directory: %PROJECT_DIR%"

cd /d "%PROJECT_DIR%"

REM ── Parse Arguments ─────────────────────────────────
set "BUILD=1"
:parse_args
if "%~1"=="" goto :done_parse
if "%~1"=="--no-build" set "BUILD=0"
if "%~1"=="--verbose" set "VERBOSE=1"
shift
goto :parse_args
:done_parse

REM ── Create Directories ──────────────────────────────
call :info "Creating directories..."
if not exist "models\checkpoints" mkdir "models\checkpoints"
if not exist "models\final" mkdir "models\final"
if not exist "data\raw" mkdir "data\raw"
if not exist "data\processed" mkdir "data\processed"
if not exist "data\tokenizer" mkdir "data\tokenizer"
if not exist "logs" mkdir "logs"
call :ok "Directories created."

REM ── Build ───────────────────────────────────────────
if "!BUILD!"=="1" (
  echo.
  call :banner "Building JALLM..."
  echo.
  call :info "Running: mvn clean package"
  mvn clean package
  if !errorlevel! neq 0 (
    call :error "Build failed. Check the output above for errors."
    pause
    exit /b 1
  )
  call :ok "Build successful."

  if exist "target\jallm.jar" (
    call :ok "JAR created: target\jallm.jar"
  ) else (
    call :warn "target\jallm.jar not found."
  )
) else (
  call :info "Build skipped (--no-build)."
)

REM ── Self-Test ───────────────────────────────────────
echo.
call :banner "Running self-test..."
echo.

if exist "target\jallm.jar" (
  java -jar target\jallm.jar selftest
  if !errorlevel! equ 0 (
    echo.
    call :ok "Self-test PASSED!"
  ) else (
    call :error "Self-test FAILED."
    pause
    exit /b 1
  )
) else (
  call :warn "target\jallm.jar not found. Skipping self-test."
)

REM ── Summary ─────────────────────────────────────────
echo.
call :banner "========================================"
call :banner "  Installation Complete!"
call :banner "========================================"
echo.
echo Project:  JALLM (Java LLM From Scratch)
echo Version:  0.1-BETA
echo Java:     !JAVA_VER!
echo OS:       Windows
echo.
echo Quick start:
echo   java -jar target\jallm.jar help
echo   java -jar target\jallm.jar train data\train.txt
echo   java -jar target\jallm.jar generate models\checkpoints\model.bin "hello" 50
echo.
echo For more information, see README.md
echo.

endlocal
