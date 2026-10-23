# =====================================================================
#   GUARD AI - Auto-Starting Local LLM (Ollama) & Web Server (PowerShell)
# =====================================================================

Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host "  GUARD AI - Auto-Starting Local LLM (Ollama) & Web Server" -ForegroundColor Cyan
Write-Host "=====================================================================" -ForegroundColor Cyan

# 1. Probe port 11434
function Test-PortListening {
    param([string]$Server = "127.0.0.1", [int]$Port = 11434)
    try {
        $tcp = New-Object System.Net.Sockets.TcpClient($Server, $Port)
        if ($tcp.Connected) {
            $tcp.Close()
            return $true
        }
    } catch {}
    return $false
}

if (Test-PortListening) {
    Write-Host "[✓] Ollama server is already running on port 11434." -ForegroundColor Green
} else {
    Write-Host "[*] Ollama is not active on port 11434. Locating executable..." -ForegroundColor Yellow
    $ollamaExe = (Get-Command ollama -ErrorAction SilentlyContinue).Source
    if (-not $ollamaExe) {
        $candidates = @(
            "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe",
            "C:\Program Files\Ollama\ollama.exe",
            "C:\Program Files (x86)\Ollama\ollama.exe"
        )
        foreach ($c in $candidates) {
            if (Test-Path $c) { $ollamaExe = $c; break }
        }
    }

    if ($ollamaExe) {
        Write-Host "[*] Found Ollama: $ollamaExe" -ForegroundColor Cyan
        Write-Host "[*] Starting Ollama server in background (ollama serve)..." -ForegroundColor Yellow
        Start-Process -FilePath $ollamaExe -ArgumentList "serve" -WindowStyle Hidden
        Start-Sleep -Seconds 2
        if (Test-PortListening) {
            Write-Host "[✓] Ollama server daemon started successfully." -ForegroundColor Green
        }
    } else {
        Write-Host "[!] Ollama is not installed yet on this system." -ForegroundColor DarkYellow
        Write-Host "[*] Guard AI will seamlessly use its high-fidelity surrogate model." -ForegroundColor Gray
        Write-Host "[*] (To use real local weights, download Ollama from https://ollama.com)" -ForegroundColor Gray
    }
}

Write-Host ""
Write-Host "[*] Launching Guard AI Streamlit Web Dashboard..." -ForegroundColor Green
python -m streamlit run app.py
