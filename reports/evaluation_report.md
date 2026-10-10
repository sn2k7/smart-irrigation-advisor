# Model Evaluation Report

> **Synthetic data notice:** All metrics were computed on SYNTHETIC data produced by this project's simulator. They demonstrate that the pipeline works and allow strategies to be compared under controlled assumptions; they are NOT evidence of real-world accuracy or water savings, which require validation with field data.

- Generated: 2026-10-10T17:30:34
- Feature schema version: 1.0
- Rows after cleaning: 106080 from 30 fields
- Split: held-out fields + chronological test period + rolling-origin CV with embargo (embargo 24 h, 3 CV folds, 63522 out-of-fold rows)
- Held-out fields: F003, F013, F018, F021, F026, F029
- Irrigation decision model: **random_forest** (threshold 0.4; chosen on cross-validated F1, the FAO-56 rule is a candidate)
- Depth model: **gradient_boosting** (chosen on cross-validated MAE; ML must beat the FAO-56 formula by >= 5%)

Selection notes: random_forest beat the FAO-56 rule on cross-validated F1 (0.890 vs 0.884). gradient_boosting beat the FAO-56 depth formula on cross-validated MAE (6.64 vs 8.57 mm).

## Classification - temporal test set (later period, seen fields)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | TN | FP | FN | TP |
|---|---|---|---|---|---|---|---|---|---|
| logistic_regression | 0.928 | 0.784 | 0.997 | 0.877 | 0.994 | 8556 | 904 | 11 | 3273 |
| random_forest | 0.932 | 0.792 | 0.998 | 0.883 | 0.997 | 8600 | 860 | 7 | 3277 |
| gradient_boosting | 0.947 | 0.831 | 0.996 | 0.906 | 0.994 | 8793 | 667 | 12 | 3272 |
| rule_based_fao56 | 0.953 | 0.853 | 0.986 | 0.915 | 0.964 | 8903 | 557 | 45 | 3239 |

## Classification - unseen fields (never used for training)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | TN | FP | FN | TP |
|---|---|---|---|---|---|---|---|---|---|
| logistic_regression | 0.906 | 0.796 | 0.953 | 0.867 | 0.982 | 13248 | 1738 | 337 | 6781 |
| random_forest | 0.900 | 0.788 | 0.942 | 0.858 | 0.978 | 13187 | 1799 | 415 | 6703 |
| gradient_boosting | 0.913 | 0.831 | 0.915 | 0.871 | 0.980 | 13666 | 1320 | 608 | 6510 |
| rule_based_fao56 | 0.926 | 0.856 | 0.928 | 0.890 | 0.927 | 13871 | 1115 | 512 | 6606 |

## Classification - rolling-origin cross-validation (used for model selection)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | TN | FP | FN | TP |
|---|---|---|---|---|---|---|---|---|---|
| logistic_regression | 0.923 | 0.895 | 0.856 | 0.875 | 0.966 | 41655 | 1998 | 2872 | 16997 |
| random_forest | 0.930 | 0.876 | 0.904 | 0.890 | 0.975 | 41112 | 2541 | 1903 | 17966 |
| gradient_boosting | 0.926 | 0.888 | 0.875 | 0.881 | 0.961 | 41472 | 2181 | 2494 | 17375 |
| rule_based_fao56 | 0.928 | 0.890 | 0.879 | 0.884 | 0.915 | 41499 | 2154 | 2410 | 17459 |

`rule_based_fao56` is the transparent physics rule used by the engine when no model is
available; it is shown as a baseline the ML models must beat.

## Regression - net irrigation depth (mm), rows where irrigation was truly needed

| Model | CV MAE | Test MAE | Test RMSE | Test R2 | Unseen MAE | Unseen RMSE | Unseen R2 |
|---|---|---|---|---|---|---|---|
| ridge | 6.975 | 11.439 | 12.34 | 0.8521 | 6.751 | 9.451 | 0.9822 |
| random_forest | 6.736 | 6.426 | 7.996 | 0.9379 | 7.012 | 11.121 | 0.9753 |
| gradient_boosting | 6.641 | 7.63 | 9.425 | 0.9137 | 8.291 | 10.964 | 0.976 |
| rule_based_fao56 | 8.57 | 9.096 | 11.486 | 0.8718 | 7.034 | 11.12 | 0.9753 |

## Most influential features (random_forest, permutation importance, out-of-fold rows)

- `dr_mean_6h_ratio`: 0.1927
- `temp_max_24h`: 0.0158
- `etc_raw_ratio`: 0.0147
- `raw_taw_ratio`: 0.0129
- `temp_min_24h`: 0.0091
- `kc`: 0.0062
- `humidity_mean_24h`: 0.0047
- `growth_stage`: 0.0045
- `available_water_fraction`: 0.0034
- `hours_since_rain`: 0.0028
- `projected_depletion_ratio`: 0.0025
- `sm_std_24h_ratio`: 0.0022
- `rain_sum_72h`: 0.0022
- `humidity_pct`: 0.0014
- `sm_delta_24h_ratio`: 0.0012

## Data quality

- rows: 106080
- fields: 30
- sm_invalid_pct: 2.65
- sm_imputed_pct: 2.33
- sm_still_missing_pct: 0.32
- spikes_detected: 282
- stuck_hours_detected: 262
- out_of_range_hours: 311
