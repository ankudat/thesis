@echo off
TITLE Thesis Environment Setup
CLS

ECHO =========================================================
ECHO   THESIS SETUP
ECHO =========================================================
ECHO.

:: 1. Check Python Version
python -c "import sys; exit(1) if sys.version_info < (3, 10) else exit(0)"
IF %ERRORLEVEL% NEQ 0 (
    ECHO [ERROR] Python 3.10 or higher is required.
    PAUSE
    EXIT /B
)

:: 2. Create Virtual Environment (if missing)
IF NOT EXIST "venv" (
    ECHO [INFO] Creating venv...
    python -m venv venv
)

:: 3. Install Libraries from requirements.txt
ECHO [INFO] Installing libraries from requirements.txt...
call venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt

ECHO.
ECHO [SUCCESS] Environment is ready!
ECHO.
PAUSE