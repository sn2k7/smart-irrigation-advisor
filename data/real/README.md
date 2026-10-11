# Put your REAL dataset here

PLACEHOLDER(real-dataset): copy your real sensor file into this folder, e.g.

    data/real/real_dataset.csv

Files in this folder are git-ignored (your data stays on your machine); only this README is tracked.

Then:
1. Edit `config/real_dataset.json`. Every value marked PLACEHOLDER must be replaced with your column names, units and field details.
2. Check the mapping:  `python scripts/train_real_dataset.py --check-only`
3. Train:              `python scripts/train_real_dataset.py`
   or use the **Real dataset** page in the desktop app (`python -m irrigation_gui`).

What the file must contain (one row per reading; hourly or more frequent is best):

| Needed | Meaning | Typical source column |
|---|---|---|
| yes | date/time of the reading | `timestamp`, `time`, `date_time` |
| yes | soil moisture (% VWC, 0-1 fraction or raw sensor counts) | `soil_moisture`, `moisture`, `vwc` |
| recommended | air temperature | `temperature`, `temp` |
| recommended | relative humidity | `humidity`, `rh` |
| recommended | rainfall since the previous row (or a running total) | `rain`, `precipitation` |
| optional | field / plot / sensor-node id (several fields per file) | `field_id`, `plot`, `node` |
| optional | your own irrigation labels (0/1) and depth (mm) | `irrigation_needed`, `irrigation_mm` |

Crop, soil, irrigation method, planting date and latitude can be columns in the file or values in the config.
