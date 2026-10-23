@echo off
setlocal enabledelayedexpansion

echo =====================================================================
echo   GUARD AI - Auto-Starting Local LLM (Ollama) & Web Server
echo =====================================================================

:: 1. Check if Ollama is running on port 11434
echo [*] Checking local Ollama service status...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "try { $tcp = New-Object System.Net.Sockets.TcpClient('127.0.0.1', 11434); if ($tcp.Connected) { $tcp.Close(); exit 0 } else { exit 1 } } catch { exit 1 }"

if %ERRORLEVEL% equ 0 (
    echo [OK] Ollama is already running on port 11434.
) else (
    echo [*] Ollama is not active on port 11434. Checking for Ollama executable...
    set OLLAMA_EXE=
    where ollama >nul 2>nul
    if !ERRORLEVEL! equ 0 (
        for /f "delims=" %%I in ('where ollama') do (
            if not defined OLLAMA_EXE set OLLAMA_EXE=%%I
        )
    )
    if not defined OLLAMA_EXE (
        if exist "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" (
            set "OLLAMA_EXE=%LOCALAPPDATA%\Programs\Ollama\ollama.exe"
        ) else if exist "C:\Program Files\Ollama\ollama.exe" (
            set "OLLAMA_EXE=C:\Program Files\Ollama\ollama.exe"
        )
    )

    if defined OLLAMA_EXE (
        echo [*] Found Ollama at: !OLLAMA_EXE!
        echo [*] Turning on Ollama background daemon (ollama serve)...
        start "" /B "!OLLAMA_EXE!" serve >nul 2>&1
        timeout /t 2 /nobreak >nul
        echo [OK] Ollama server daemon started.
    ) else (
        echo [!] Note: Ollama is not yet installed on this system.
        echo [*] Guard AI will seamlessly use its built-in high-fidelity surrogate model.
        echo [*] (To use real local LLM weights, install Ollama from https://ollama.com)
    )
)

echo.
echo [*] Starting Guard AI Streamlit Web Dashboard...
python -m streamlit run app.py

pause
