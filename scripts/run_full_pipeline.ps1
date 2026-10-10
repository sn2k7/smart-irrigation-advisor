# Smart Irrigation Advisor - full reproducible pipeline (Windows PowerShell)
# Run from the project root with the virtual environment activated:
#   .\scripts\run_full_pipeline.ps1            # full run (~5-10 min)
#   .\scripts\run_full_pipeline.ps1 -Fast      # quick run (~2 min)
param([switch]$Fast)
$ErrorActionPreference = "Stop"

function Step($text) { Write-Host "`n=== $text ===" -ForegroundColor Cyan }

Step "1/5 Generate SYNTHETIC dataset"
python scripts/generate_dataset.py --fields 30 --seed 42
if ($LASTEXITCODE -ne 0) { throw "dataset generation failed" }

Step "2/5 Train and evaluate models"
if ($Fast) { python scripts/train_models.py --fast } else { python scripts/train_models.py }
if ($LASTEXITCODE -ne 0) { throw "training failed" }

Step "3/5 Closed-loop strategy simulation"
if ($Fast) { python scripts/run_simulation.py --fields 4 } else { python scripts/run_simulation.py --fields 10 }
if ($LASTEXITCODE -ne 0) { throw "simulation failed" }

Step "4/5 Seed demo database"
python scripts/seed_demo_data.py --reset
if ($LASTEXITCODE -ne 0) { throw "seeding failed" }

Step "5/5 Run tests"
python -m pytest -q
if ($LASTEXITCODE -ne 0) { throw "tests failed" }

Write-Host "`nDone. Reports are in .\reports. Start the API with:" -ForegroundColor Green
Write-Host "  uvicorn irrigation_advisor.api.main:app --host 127.0.0.1 --port 8000"
Write-Host "  then open http://127.0.0.1:8000/docs"
