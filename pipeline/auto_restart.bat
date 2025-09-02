@echo off
REM Remove old "data" folder if it exists
IF EXIST data (
    rmdir /S /Q data
)

REM Copy "raw" to "data" silently (no output)
xcopy raw data /E /I /Y >nul 2>&1

REM Run Python script
python main.py
