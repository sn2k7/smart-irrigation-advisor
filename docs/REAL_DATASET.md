# Using a REAL dataset

> Ready-made real source: **Open-Meteo** data for 11.93975 N, 79.49244 E. See [OPENMETEO.md](OPENMETEO.md).

Everything you must adapt is marked **`PLACEHOLDER(real-dataset)`**. To list every spot:

```powershell
Select-String -Path config\*.json, irrigation_advisor\**\*.py, scripts\*.py, *.py -Pattern "PLACEHOLDER(real-dataset)" -SimpleMatch
```

| File | What to fill in |
|---|---|
| `data/real/` | put your file here (git-ignored), see `data/real/README.md` for the required columns |
| `config/real_dataset.json` | path, your column names, units, timestamp format, crop/soil/method/planting date/latitude, label mode. Every key has a `_PLACEHOLDER_...` comment. Unfilled `<...>` values are rejected with the exact key name. |
| `irrigation_advisor/data/dataset_hooks.py` | optional code for merging files, pivoting long data, per-probe calibration or building labels |
| `irrigation_advisor/real_dataset_workflow.py` | defaults: config path, model output path, reports folder, minimum data requirements |

## Frontend: Real Dataset Studio (PySide6)

`real_dataset_gui.py` is a standalone desktop app with three tabs:

1. **Configure**: browse to your file, map its columns from drop-downs, choose units and field details,
   and save. The `_PLACEHOLDER` comments in the JSON are kept.
2. **Check**: converts the file exactly as training will. It shows READY / NOT READY, problems, warnings,
   rows/fields/time span/interval, the share of flagged readings, a soil-moisture chart and a preview.
3. **Train**: leakage-safe training and evaluation. It shows which model was deployed and why, and test
   metrics for every candidate (including the FAO-56 rule). It can also ask the running API to reload
   the new model.

```powershell
.\.venv\Scripts\Activate.ps1
.\scripts\run_real_dataset_gui.ps1         # or: python real_dataset_gui.py
```

Screenshots (stand-in data, not a real farm): `docs/screenshots/real_studio_configure.png`,
`real_studio_check.png`, `real_studio_train.png`.

It is a desktop program, so it has no web address. It runs on the computer that holds the data and the
project code.

## Command line

```powershell
python scripts/train_real_dataset.py --check-only     # validate only
python scripts/train_real_dataset.py --fast           # quick first training
python scripts/train_real_dataset.py                  # final training
```

## Requirements before training (editable in `real_dataset_workflow.py`)
* at least about two weeks of hourly data (`MIN_ROWS`, `MIN_DAYS`);
* soil moisture usable for at least 70 % of hours after cleaning;
* with your own labels: both "irrigate" and "don't irrigate" cases.

## Honest evaluation
With `labels.mode = "derive"`, labels are computed from your observations with the FAO-56 water balance,
so the rule baseline partly grades itself (expect near-perfect rule scores). For an independent result,
supply real labels (`"columns"`), e.g. agronomist decisions or observed crop stress.
