# Start the Real Dataset Studio (PySide6) - run from the project root with the venv active.
#   .\scripts\run_real_dataset_gui.ps1
#   .\scripts\run_real_dataset_gui.ps1 -Config config\my_other_dataset.json
param([string]$Config = "config\real_dataset.json", [string]$ApiUrl = "http://127.0.0.1:8000")
$ErrorActionPreference = "Stop"
python real_dataset_gui.py --config $Config --api-url $ApiUrl
