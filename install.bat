@echo off
REM Installation script for PDF Analyzer CLI (Windows)
REM Supports Florence-2 and any Hugging Face vision model
REM Uses uv for dependency management

setlocal enabledelayedexpansion

echo.
echo 🚀 AI Drawing Analyzer - Installation (Windows)
echo ======================================================
echo.
echo This script will help you set up the PDF Analyzer for:
echo   ✅ Florence-2 OCR (local, no API key^)
echo   ✅ Any Hugging Face vision model locally
echo   ✅ Cloud APIs (Google Gemini, OpenAI, Anthropic, etc.^)
echo.

REM Check Python installation and version
set "PYTHON_CMD="
set "PYTHON_OK=0"
set "MIN_MAJOR=3"
set "MIN_MINOR=9"

REM Try different Python commands (python, python3, py)
for %%P in (python python3 py) do (
    if "!PYTHON_CMD!"=="" (
        %%P --version >nul 2>&1
        if !errorlevel! equ 0 (
            set "PYTHON_CMD=%%P"
        )
    )
)

if "!PYTHON_CMD!"=="" (
    echo ❌ Python is not installed or not in PATH
    goto :install_python_prompt
)

REM Get Python version and parse it
for /f "tokens=2 delims= " %%V in ('!PYTHON_CMD! --version 2^>^&1') do set "PYTHON_VERSION=%%V"
echo    Detected Python version: !PYTHON_VERSION!

REM Parse major and minor version numbers
for /f "tokens=1,2 delims=." %%A in ("!PYTHON_VERSION!") do (
    set "PY_MAJOR=%%A"
    set "PY_MINOR=%%B"
)

REM Check if version is >= 3.9
if !PY_MAJOR! gtr !MIN_MAJOR! (
    set "PYTHON_OK=1"
) else if !PY_MAJOR! equ !MIN_MAJOR! (
    if !PY_MINOR! geq !MIN_MINOR! (
        set "PYTHON_OK=1"
    )
)

if "!PYTHON_OK!"=="0" (
    echo.
    echo ❌ Python version !PYTHON_VERSION! is too old.
    echo    This project requires Python !MIN_MAJOR!.!MIN_MINOR! or higher.
    goto :install_python_prompt
)

echo [OK] Python !PYTHON_VERSION! found (using !PYTHON_CMD!^)
echo.
goto :python_ok

:install_python_prompt
echo.
echo ════════════════════════════════════════════════════════════
echo Python !MIN_MAJOR!.!MIN_MINOR! or higher is required but not found.
echo ════════════════════════════════════════════════════════════
echo.
echo Choose an option:
echo   1^) Download and install Python automatically (recommended^)
echo   2^) Open python.org in browser to download manually
echo   3^) Exit and install Python yourself
echo.
set /p INSTALL_CHOICE="Enter choice (1-3): "

if "!INSTALL_CHOICE!"=="1" goto :auto_install_python
if "!INSTALL_CHOICE!"=="2" goto :open_python_website
if "!INSTALL_CHOICE!"=="3" goto :exit_no_python
goto :install_python_prompt

:auto_install_python
echo.
echo 📥 Downloading Python installer...
echo    This will download Python 3.12 (latest stable^)
echo.

REM Create temp directory for download
set "TEMP_DIR=%TEMP%\python_install"
if not exist "!TEMP_DIR!" mkdir "!TEMP_DIR!"

REM Detect system architecture
set "ARCH=amd64"
if "%PROCESSOR_ARCHITECTURE%"=="x86" (
    if not defined PROCESSOR_ARCHITEW6432 (
        set "ARCH=win32"
    )
)

REM Download Python installer using PowerShell
set "PYTHON_URL=https://www.python.org/ftp/python/3.12.7/python-3.12.7-!ARCH!.exe"
set "INSTALLER_PATH=!TEMP_DIR!\python-installer.exe"

echo    Downloading from: !PYTHON_URL!
echo.

powershell -Command "& {[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '!PYTHON_URL!' -OutFile '!INSTALLER_PATH!' -UseBasicParsing}"

if not exist "!INSTALLER_PATH!" (
    echo ❌ Failed to download Python installer.
    echo    Please download manually from https://www.python.org/downloads/
    pause
    exit /b 1
)

echo ✅ Download complete!
echo.
echo 🔧 Installing Python 3.12...
echo    • Adding Python to PATH
echo    • Installing pip
echo    • Installing for current user
echo.
echo    Please wait... (this may take 1-2 minutes^)
echo.

REM Run installer with options: add to PATH, install pip, install for user
"!INSTALLER_PATH!" /passive InstallAllUsers=0 PrependPath=1 Include_pip=1 Include_launcher=1

if errorlevel 1 (
    echo ❌ Python installation failed.
    echo    Please try installing manually from https://www.python.org/downloads/
    pause
    exit /b 1
)

echo.
echo ✅ Python installed successfully!
echo.
echo ⚠️  IMPORTANT: Please close this window and run install.bat again
echo    to use the newly installed Python.
echo.
del "!INSTALLER_PATH!" 2>nul
pause
exit /b 0

:open_python_website
echo.
echo 🌐 Opening python.org downloads page...
start https://www.python.org/downloads/
echo.
echo After installing Python:
echo   1. Make sure to check "Add Python to PATH" during installation
echo   2. Close and reopen this terminal
echo   3. Run install.bat again
echo.
pause
exit /b 0

:exit_no_python
echo.
echo Installation cancelled. Please install Python !MIN_MAJOR!.!MIN_MINOR!+ from:
echo   https://www.python.org/downloads/
echo.
echo Make sure to check "Add Python to PATH" during installation.
pause
exit /b 1

:python_ok

REM Check and install uv if needed
uv --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo 📦 Installing uv (Python package manager)...
    powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    uv --version >nul 2>&1
    if errorlevel 1 (
        echo ❌ Failed to install uv. Please install manually from https://astral.sh/uv
        pause
        exit /b 1
    )
    echo ✅ uv installed successfully
) else (
    for /f "tokens=*" %%i in ('uv --version') do set UV_VERSION=%%i
    echo [OK] Found uv: !UV_VERSION!
)
echo.

REM Ask user for installation type
echo Choose your installation type:
echo ════════════════════════════════════════════════════════════
echo 1️⃣  LOCAL MODELS ONLY (Recommended^)
echo    • Run Florence-2, Qwen-VL, Qwen3-VL, LLaVA locally
echo    • No API key required
echo    • Best for: Technical drawings, blueprints
echo.
echo 2️⃣  CLOUD APIs ONLY (Minimal^)
echo    • Use Gemini, OpenAI, Claude, etc.
echo    • Requires API keys
echo    • Best for: Quick testing, occasional use
echo.
echo 3️⃣  BOTH LOCAL and CLOUD (Full Setup^)
echo    • All features: local models + cloud APIs
echo    • Choose which to use at runtime
echo ════════════════════════════════════════════════════════════
set /p choice="Enter choice (1-3, default: 1): "
if "!choice!"=="" set choice=1

REM Install dependencies based on choice using uv
echo.
echo 📥 Installing dependencies with uv...
echo    (This may take 2-5 minutes depending on your internet^)
echo.

if "!choice!"=="1" (
    echo 📚 Installing LOCAL MODEL SUPPORT...
    echo    • transformers, torch, timm, einops
    uv sync --extra local
    echo.
    echo 🚀 GPU ACCELERATION (Optional but Recommended^)
    echo    To use GPU (CUDA 11.8^) for 10-100x faster inference:
    echo.
    echo    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    echo.
    echo    For other CUDA versions, visit: https://pytorch.org/get-started/locally/
    echo.
) else if "!choice!"=="2" (
    echo ☁️  Installing CLOUD API SUPPORT...
    echo    • httpx, pymupdf, Pillow, google-auth, etc.
    uv sync
    echo.
) else if "!choice!"=="3" (
    echo 🔗 Installing FULL SETUP (Local + Cloud^)...
    echo    • All local model dependencies
    echo    • All cloud API dependencies
    uv sync --all-extras
    echo.
    echo 🚀 GPU ACCELERATION (Optional but Recommended^)
    echo    To use GPU (CUDA 11.8^) for faster inference:
    echo.
    echo    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    echo.
) else (
    echo ❓ Invalid choice. Using LOCAL MODELS (option 1^) by default...
    uv sync --extra local
)

REM Create .env from example if it doesn't exist
if not exist .env (
    echo.
    echo 📝 Creating .env file from template...
    if exist .env.example (
        copy .env.example .env
        if "!choice!"=="1" (
            echo [OK] .env file created (not needed for local models^)
        ) else (
            echo [OK] .env file created. Configure API keys to use cloud providers.
        )
    )
)

echo.
echo ======================================================
echo ✅ Installation Complete!
echo ======================================================
echo.
echo 🎯 NEXT STEPS:
echo.

if "!choice!"=="1" (
    echo 1️⃣  Activate environment:
    echo    .venv\Scripts\activate.bat
    echo.
    echo 2️⃣  Test with Florence-2 (interactive^):
    echo    ai-drawing-analyzer your_document.pdf
    echo.
    echo 3️⃣  Or use command-line directly:
    echo    ai-drawing-analyzer doc.pdf -p huggingface-local -m microsoft/Florence-2-large
    echo.
    echo 📊 First run will download the model (~2-24GB, takes 5-10 minutes^)
    echo    Model is cached afterwards for fast reuse
) else if "!choice!"=="2" (
    echo 1️⃣  Activate environment:
    echo    .venv\Scripts\activate.bat
    echo.
    echo 2️⃣  Add API keys to .env file:
    echo    notepad .env
    echo.
    echo 3️⃣  Test with Gemini (free tier^):
    echo    ai-drawing-analyzer your_document.pdf -p gemini
    echo.
    echo 🆓 Get free API keys:
    echo    • Google Gemini: https://makersuite.google.com/app/apikey
    echo    • HuggingFace Router: https://huggingface.co/settings/tokens
) else (
    echo 1️⃣  Activate environment:
    echo    .venv\Scripts\activate.bat
    echo.
    echo 2️⃣  Choose your path:
    echo.
    echo    LOCAL (no API key^):
    echo    ai-drawing-analyzer doc.pdf -p huggingface-local
    echo.
    echo    CLOUD (with API key^):
    echo    notepad .env  (add your API keys^)
    echo    ai-drawing-analyzer doc.pdf -p gemini
    echo.
)

echo.
echo 📖 Documentation:
echo    • Quick Start: QUICK_START.md
echo    • Full Guide: README.md
echo    • Help: ai-drawing-analyzer --help
echo.
echo ======================================================
echo.
pause
