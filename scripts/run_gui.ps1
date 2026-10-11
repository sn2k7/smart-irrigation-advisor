# Start the API (minimised window) and the PySide6 desktop app (Windows PowerShell).
# Run from the project root with the virtual environment activated:
#   .\scripts\run_gui.ps1                 # starts API + GUI
#   .\scripts\run_gui.ps1 -NoServer       # GUI only (API already running / on another PC)
param([string]$ApiUrl = "http://127.0.0.1:8000", [switch]$NoServer)
$ErrorActionPreference = "Stop"

if (-not $NoServer) {
    Write-Host "Starting API on $ApiUrl ..." -ForegroundColor Cyan
    $port = ([uri]$ApiUrl).Port
    Start-Process -FilePath "python" -ArgumentList "-m","uvicorn","irrigation_advisor.api.main:app","--host","127.0.0.1","--port",$port -WindowStyle Minimized
    for ($i = 0; $i -lt 30; $i++) {
        try { Invoke-RestMethod "$ApiUrl/health" | Out-Null; break } catch { Start-Sleep -Seconds 1 }
    }
}
Write-Host "Starting desktop app ..." -ForegroundColor Cyan
python -m irrigation_gui --api-url $ApiUrl
