@echo off
TITLE Thesis Environment Setup
CLS

:: 0. Force the script to run in its own directory (Fixes System32 issue)
cd /d "%~dp0"

ECHO =========================================================
ECHO   THESIS SETUP AND RESET
ECHO =========================================================
ECHO.

:: 1. Check Python Version
python -c "import sys; exit(1) if sys.version_info < (3, 10) else exit(0)"
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
        ECHO [INFO] Nusing old environment...
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
pip install -r requirements.txt

:: 5. Download the Pinned spaCy Model
ECHO [INFO] Downloading exact German spaCy model...
python -m spacy download de_core_news_lg-3.8.0 --direct

ECHO.
ECHO [SUCCESS] Environment is perfectly configured and ready!
ECHO.
PAUSE