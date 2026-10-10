# Irrigation Strategy Comparison (Closed-Loop Simulation)

> **SIMULATED RESULTS - produced by a closed-loop soil water balance model with synthetic weather and sensor noise. They compare strategies under identical, documented assumptions and are NOT measured real-world water savings. Field trials are required before claiming savings.**

- Virtual fields: 10 (seed 2024, different from the training data)
- Fixed schedules irrigate every 3 days with a depth sized for peak demand
- Sensor faults injected: yes

## Mean per field and season

Baselines (no sensors): `fixed_schedule` (pure calendar, depth for peak demand), `fixed_skip_after_rain` (same, skipped after >= 10 mm rain) and `calendar_kc_adjusted` (depth follows the crop-stage Kc curve, skipped after rain).

| Strategy | Water (m³/ha) | Events | Pump h | Stress days | Severe stress days | Rel. transpiration | Drainage (mm) |
|---|---|---|---|---|---|---|---|
| fixed_schedule | 12724 | 50.9 | 150.1 | 0.3 | 0.0 | 1.000 | 780 |
| fixed_skip_after_rain | 11292 | 45.3 | 133.9 | 0.3 | 0.0 | 1.000 | 683 |
| calendar_kc_adjusted | 8516 | 45.3 | 104.2 | 6.1 | 0.0 | 0.990 | 496 |
| advisor_rules | 4643 | 12.3 | 59.1 | 5.1 | 0.0 | 0.998 | 258 |
| advisor_ml_hybrid | 4537 | 12.4 | 59.9 | 7.2 | 0.0 | 0.997 | 248 |

## Versus the pure fixed schedule (ignores rain)

- **advisor_rules**: simulated water use +63.5% saving overall (median field +61.5%, range +34.6% to +91.8%, 10/10 fields saved water); stress days +4.84; relative transpiration -0.0015; drainage -521 mm
- **advisor_ml_hybrid**: simulated water use +64.3% saving overall (median field +64.0%, range +33.0% to +91.8%, 10/10 fields saved water); stress days +6.91; relative transpiration -0.0027; drainage -532 mm

## Versus the fixed schedule that skips after >= 10 mm rain

- **advisor_rules**: simulated water use +58.9% saving overall (median field +56.3%, range +34.6% to +89.2%, 10/10 fields saved water); stress days +4.84; relative transpiration -0.0015; drainage -425 mm
- **advisor_ml_hybrid**: simulated water use +59.8% saving overall (median field +60.4%, range +33.0% to +89.2%, 10/10 fields saved water); stress days +6.91; relative transpiration -0.0027; drainage -435 mm

## Versus the crop-stage (Kc) adjusted calendar that also skips after rain (strongest no-sensor baseline)

- **advisor_rules**: simulated water use +45.5% saving overall (median field +41.5%, range +10.1% to +85.1%, 10/10 fields saved water); stress days -0.95; relative transpiration +0.0083; drainage -238 mm
- **advisor_ml_hybrid**: simulated water use +46.7% saving overall (median field +47.4%, range +7.9% to +85.1%, 10/10 fields saved water); stress days +1.12; relative transpiration +0.0071; drainage -248 mm

## How to read these numbers

- The size of the 'saving' depends strongly on the baseline. Calendar schedules sized for peak demand
  over-irrigate early and late in the season and during rain (see the drainage column); the
  crop-stage adjusted calendar is the most demanding comparison and gives the most conservative figure.
- Stress days count hours with Ks < 1, i.e. depletion beyond RAW. The advisor deliberately lets the root
  zone dry to the RAW threshold before refilling, so short mild stress is expected; relative
  transpiration (actual / potential crop ET, a yield proxy) shows its size.
- Weather, soil physics, sensor noise and faults are simulated; real savings must be measured in field trials.
