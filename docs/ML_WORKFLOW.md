# Machine-learning workflow

> Everything in this document refers to **synthetic data** produced by this project's simulator.
> The metrics show that the pipeline works and is evaluated correctly. They are not evidence of
> real-world accuracy, which needs field data.

## 1. Problem definition

At decision time *t*, for one field:

| Task | Target | Definition (`data/labels.py`) |
|---|---|---|
| Classification | `target_irrigation_needed` | 1 if, **without irrigation**, root-zone depletion reaches RAW (onset of crop water stress) at *t* or within the next 24 h, given the rain that actually falls |
| Regression (optional) | `target_irrigation_mm` | net depth needed when irrigation is needed: refill to field capacity, credit effective rain in the next 24 h, at least one day of crop ET, at most TAW |

Duration and water volume are **not** learned. They follow deterministically from the net depth:
`gross = net / efficiency`, `litres = gross × area`, `minutes = litres / flow_rate`. This keeps the ML
problem independent of field size and pump, and keeps the arithmetic auditable.

## 2. Data

### Synthetic generator (`data/simulator.py`)
* 30 fields (seed 42) with random crop, soil, irrigation method, planting date (spread over a year),
  latitude, area, flow rate and **sensor calibration bias**.
* Weather: Markov-chain rain occurrence + gamma amounts with a monsoon season, AR(1) temperature anomalies,
  diurnal temperature/humidity cycles, afternoon storms (`data/climate.py`).
* "True" soil water: hourly FAO-56 bucket with interception, runoff, stress-reduced ET and drainage
  (`agronomy/soil_model.py`). The true ET0 includes humidity and cloud effects that the model's Hargreaves
  estimate does not know about - as in reality, the features are an imperfect view of the process.
* Historical farmer behaviour: calendar irrigators (some skip after rain, some miss days) and "visual"
  irrigators with noisy perception of dryness - this creates both well-watered and stressed periods.
* Sensors: bias + Gaussian noise + 0.1 % resolution, DHT22-like temperature/humidity noise, 0.2 mm
  tipping-bucket rain; injected faults: dropouts, multi-hour outages, spikes, stuck values, impossible
  values (e.g. 99.9 % or 85 °C).
* Rain forecast: actual next-24 h rain with 12 % misses, 8 % false alarms and log-normal error.
* Labels: counterfactual 24 h no-irrigation projection from the **true** soil state.

### External CSV (`data/loader.py`)
Column aliases (`time`, `moisture`, `temp`, `rain`, `plot_id`, ...) are mapped to the internal schema;
missing metadata gets explicit defaults. Without labels, **weak labels** are derived from observations
(sensor depletion + observed rain + Hargreaves ETc of the next 24 h). Run:
`python scripts/train_models.py --external-csv my.csv --crop maize --soil loam`.

## 3. Identical preprocessing and features (training = inference)

```
raw readings ─► prepare_sensor_frame ─► build_features ─► FEATURE_COLUMNS ─► sklearn Pipeline
```

* `prepare_sensor_frame` (`data/preprocessing.py`): type coercion, timestamp normalisation, de-duplication,
  physical range checks, hourly resampling per field, backward-looking spike/stuck detection, forward fill
  of gaps up to 3 h with an imputation flag. **No linear interpolation** (it would use the next reading).
* `build_features` (`data/features.py`): per field, only past rows (`rolling`, `shift(+k)`), plus the rain
  forecast available at decision time.
* Imputation, scaling and one-hot encoding live inside the saved sklearn `Pipeline`, fitted on training rows.
* The saved bundle records `FEATURE_SCHEMA_VERSION` and the exact column list; loading a mismatching model
  is refused and the engine falls back to rules.
* Tests: `test_features_do_not_use_future_data` (randomising every row after *t* leaves all features at
  ≤ *t* unchanged) and `test_truncated_history_gives_identical_last_row` (the 72 h inference window gives
  the same features as the full training series).

### Feature set (26 model inputs, all dimensionless or weather values)

| Group | Features |
|---|---|
| Water status vs. stress threshold | `depletion_ratio` (Dr/RAW), `dr_mean_6h_ratio`, `projected_depletion_ratio` ((Dr + ETc − effective forecast rain)/RAW), `available_water_fraction`, `raw_taw_ratio` (= p), `kc`, `growth_stage` |
| Demand and supply vs. RAW | `etc_raw_ratio`, `forecast_rain_raw_ratio`, `rain_24h_raw_ratio` |
| Moisture dynamics | `sm_delta_{1h,6h,24h}_ratio`, `sm_std_24h_ratio` (% VWC changes converted to mm over the root depth, / RAW) |
| Weather history | `temperature_c`, `humidity_pct`, `rainfall_mm`, `temp_max_24h`, `temp_min_24h`, `humidity_mean_24h`, `rain_sum_72h`, `hours_since_rain` |
| Time / data quality | `hour_sin`, `hour_cos`, `sm_imputed`, `sm_invalid_hours_24h` |

Absolute magnitudes (mm, % VWC), day-of-year and crop/soil identity are computed (`CONTEXT_COLUMNS`) for
the rule and explanations but are **not** model inputs - see the ablation in section 6.

## 4. Leakage prevention and evaluation design (`ml/splitting.py`)

```
fields ──┬── 80 % "seen" fields ──► time ─────────────────────────────────────────────►
         │                          [ development period ][24 h embargo][ temporal TEST ]
         │                           ├ fold1: train│gap│val
         │                           ├ fold2: train ──────│gap│val
         │                           └ fold3: train ─────────────│gap│val
         └── 20 % held-out fields ──► unseen-field TEST (never used before final evaluation)
```

* Labels look 24 h ahead, so every train→evaluation boundary has a **24 h embargo**; `assert_no_leakage`
  checks this for every fold and both test sets.
* No random row-wise splits (adjacent hours are near-duplicates).
* Model selection and threshold tuning use **only out-of-fold predictions** from rolling-origin CV.
* Selected models are refitted on the whole development period and evaluated **once** on the test sets.

## 5. Models and selection (`ml/pipelines.py`, `ml/training.py`)

Candidates for the decision: Logistic Regression (balanced class weights), Random Forest
(250 trees, depth 14, balanced subsample), Gradient Boosting (200 trees, depth 3) - **and the FAO-56 rule**.
Candidates for depth: Ridge, Random Forest, Gradient Boosting regressors trained on `depth / RAW`
(dimensionless target, rescaled at prediction) - **and the FAO-56 refill formula**.

Selection rules (fixed before the final test evaluation):
* decision: highest cross-validated F1; ML is deployed only if it beats the rule's CV F1; threshold =
  the F1-optimal threshold on out-of-fold predictions;
* depth: lowest cross-validated MAE; ML must beat the formula by ≥ 5 % (parsimony: depth errors cost water
  directly and the formula is fully explainable).

## 6. Results (synthetic; full tables in `reports/evaluation_report.md`)

Selected: **Random Forest** for the decision (CV F1 0.890 vs rule 0.884, threshold 0.40) and
**Gradient Boosting** for depth (CV MAE 6.64 mm vs formula 8.57 mm).

| Decision model | CV F1 | Temporal test F1 | Unseen fields F1 | Temporal test recall | ROC-AUC (test) |
|---|---|---|---|---|---|
| Logistic Regression | 0.875 | 0.877 | 0.867 | 0.997 | 0.994 |
| Random Forest (selected) | 0.890 | 0.883 | 0.858 | 0.998 | 0.997 |
| Gradient Boosting | 0.881 | 0.906 | 0.871 | 0.996 | 0.994 |
| FAO-56 rule | 0.884 | 0.915 | 0.890 | 0.986 | 0.964 |

| Depth model | CV MAE (mm) | Temporal test MAE | Unseen fields MAE |
|---|---|---|---|
| Ridge | 6.98 | 11.44 | 6.75 |
| Random Forest | 6.74 | 6.43 | 7.01 |
| Gradient Boosting (selected) | 6.64 | 7.63 | 8.29 |
| FAO-56 formula | 8.57 | 9.10 | 7.03 |

**Interpretation - say this in the viva:**
* On data generated *from* FAO-56 physics, the physics rule is a very strong baseline. The ML classifiers
  are within 0.01 F1 of it in cross-validation and 0.01-0.04 F1 below it on the test sets. They have higher
  recall (fewer missed irrigation needs) and better ranking (ROC-AUC 0.99 vs 0.96), but lower precision at
  the chosen threshold. The selected Random Forest is *worse* than the rule on both test sets (F1 0.883 vs
  0.915 and 0.858 vs 0.890): CV-based selection does not guarantee test superiority, and we report that
  rather than re-tuning on the test set.
* The ML depth regressor improves on the formula on the temporal test (−1.5 mm MAE for the selected model),
  mainly by correcting the Hargreaves ET bias and smoothing sensor noise, but not on unseen fields.
* This is why the engine is a **hybrid**: the rule is always computed, disagreements are surfaced and
  lower the confidence, and the safety rules apply regardless of the model.
* On real data the balance can shift either way. The method (CV selection with the rule as a candidate)
  stays the same.

### Development history (disclosed for transparency)
1. The first feature set used absolute values (mm, % VWC, day-of-year, crop/soil one-hot). It overfitted
   individual fields and seasons: the FAO-56 rule beat every ML model on held-out data. An ablation showed
   that dimensionless, RAW-normalised features generalise much better, so they replaced the absolute ones.
2. Model selection first used a single validation window. It picked a model that did worse on the
   temporal test than an alternative did, a sign that one short window is unstable across seasons. It
   was replaced by rolling-origin CV over the whole development period. **This change was made after
   looking at test results, so the reported test numbers are slightly optimistic for the procedure as a
   whole.** A fresh dataset (`--seed`) gives an untouched evaluation.

### Most influential features
Permutation importance (out-of-fold rows) is dominated by `dr_mean_6h_ratio` (smoothed depletion relative
to RAW), followed by temperature/ETc terms and `raw_taw_ratio`. The model mostly relies on the same physics
as the rule, plus weather context. That is plausible and easy to explain.

## 7. Reproducing

```powershell
python scripts/generate_dataset.py --fields 30 --seed 42
python scripts/train_models.py            # ~1-3 min; writes models/ and reports/
python scripts/train_models.py --fast     # quick variant (fewer trees)
```
Seeds are fixed (data, splits, models), so the numbers above are reproduced exactly with the same
library versions; small differences can appear with other scikit-learn/numpy versions.

## 8. Limitations
* Synthetic labels are derived from the same FAO-56 physics as the rule, which favours the rule.
* Sensor calibration bias per probe cannot be learned from a single probe's data; calibrate probes.
* Hargreaves ET0 ignores wind and radiation; a station with those sensors should use Penman-Monteith.
* Paddy rice and perennial crops are out of scope.
* Field validation (e.g. a season of paired plots) is required before any agronomic claim.
