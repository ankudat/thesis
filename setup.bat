@echo off
TITLE Thesis Environment Setup
CLS

:: 0. Force the script to run in its own directory (Fixes System32 issue)
cd /d "%~dp0"

ECHO =========================================================
ECHO   THESIS SETUP AND RESET (RTX 4090 EDITION)
ECHO =========================================================
ECHO.

:: 1. Check Python Version
python -c "import sys; sys.exit(1) if sys.version_info < (3, 10) else sys.exit(0)"
IF %ERRORLEVEL% NEQ 0 (
    ECHO [ERROR] Python 3.10 or higher is required.
    PAUSE
    EXIT /B
)

:: 2. Reset Environment Check
IF EXIST "venv" (
    ECHO [INFO] Existing 'venv' found.
    SET /P AREYOUSURE="Do you want to completely delete and reset it? (Y/[N])? "
    IF /I "%AREYOUSURE%"=="Y" (
        ECHO [INFO] Nuking old environment...
        rmdir /s /q venv
    )
)

:: 3. Create Virtual Environment
IF NOT EXIST "venv" (
    ECHO [INFO] Creating fresh venv...
    python -m venv venv
)

:: 4. Install Libraries
ECHO [INFO] Activating environment and installing libraries...
call venv\Scripts\activate
python -m pip install --upgrade pip

:: Force CUDA PyTorch installation for Windows before resolving requirements.txt
ECHO [INFO] Installing PyTorch with CUDA 12.4 support for RTX 4090...
pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

ECHO [INFO] Installing remaining dependencies from requirements.txt...
pip install -r requirements.txt

:: 5. Download the spaCy Model
ECHO [INFO] Downloading German spaCy model...
python -m spacy download de_core_news_lg

ECHO.
ECHO =========================================================
ECHO [SUCCESS] Environment is configured and ready!
ECHO.
ECHO [ACTION REQUIRED]
ECHO If you haven't authenticated with HuggingFace yet to access Llama models:
ECHO 1. Type: venv\Scripts\activate
ECHO 2. Type: huggingface-cli login
ECHO =========================================================
ECHO.
PAUSE