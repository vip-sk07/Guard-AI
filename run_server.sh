#!/usr/bin/env bash
# =====================================================================
#   GUARD AI - Auto-Starting Local LLM (Ollama) & Web Server (Linux)
# =====================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "====================================================================="
echo "  GUARD AI - Auto-Starting Local LLM (Ollama) & Web Server"
echo "====================================================================="

# 1. Check if Ollama is listening on port 11434
echo "[*] Checking local Ollama service status..."
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "[OK] Ollama is already running on port 11434."
else
    echo "[*] Ollama is not active on port 11434. Checking for Ollama executable..."
    if command -v ollama > /dev/null 2>&1; then
        echo "[*] Starting Ollama daemon (ollama serve)..."
        ollama serve > /dev/null 2>&1 &
        sleep 2
        echo "[OK] Ollama server daemon started."
    else
        echo "[!] Note: Ollama is not installed or not in PATH."
        echo "[*] Guard AI will seamlessly use its built-in surrogate model."
    fi
fi

echo ""
echo "[*] Starting Guard AI Streamlit Web Dashboard..."
exec python3 -m streamlit run app.py
