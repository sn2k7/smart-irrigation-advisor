# Dataset configuration (use your own data)

| File | Purpose |
|---|---|
| `dataset_config.example.json` | **Template** - copy it, replace every `<...>` value with your column names, units and field metadata |
| `dataset_config.sample.json` | Working example for a deliberately "foreign" CSV format (different names, Fahrenheit, fraction moisture, cumulative rain in inches) |

```powershell
Copy-Item config\dataset_config.example.json config\my_dataset.json   # then edit it
python scripts/prepare_custom_dataset.py --config config\my_dataset.json   # check mapping + units
python scripts/train_models.py --dataset-config config\my_dataset.json     # train + evaluate on it
```

Anything a mapping cannot express (merging files, pivoting long data, per-probe calibration, your own
labels) goes into `irrigation_advisor/data/dataset_hooks.py` (search for `TODO(dataset)`).

Data requirements for training: hourly or finer readings, ideally several fields and several weeks,
with both irrigated/rainy and drying periods. Very small files can be checked with `prepare_custom_dataset.py`,
but they are not enough to train. Note: the sample configs leave out keys you do not need, such as `soil_raw_calibration`.

**Interpreting results with `labels.mode = "derive"`:** the weak labels are computed with the FAO-56
water balance, so the `rule_based_fao56` baseline is partly scored against its own definition and will
look near-perfect. For an independent evaluation provide real labels (`labels.mode = "columns"`), e.g.
from agronomist assessments or plant-stress observations.
