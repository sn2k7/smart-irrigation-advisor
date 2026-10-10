# Example payloads

All values are illustrative (not real field measurements).

| File | Endpoint |
|---|---|
| `field_create.json` | `POST /api/v1/fields` |
| `sensor_reading.json` | `POST /api/v1/readings` |
| `sensor_reading_faulty.json` | `POST /api/v1/readings` - stored but flagged invalid (out-of-range moisture, DHT22 error value 85 °C, low battery) |
| `sensor_batch.json` | `POST /api/v1/readings/batch` |
| `iot_payload.json` | `POST /api/v1/iot/ingest?recommend=true` (register the device first with `POST /api/v1/devices`) |
| `recommendation_request.json` | `POST /api/v1/recommendations` |
| `recommendation_request_with_reading.json` | `POST /api/v1/recommendations` (stores the reading, then decides; uses your own rain forecast) |
| `irrigation_event.json` | `POST /api/v1/fields/{id}/irrigation-events` |
| `calculator_request.json` | `POST /api/v1/tools/irrigation-calculator` |
| `sample_external_dataset.csv` | `python scripts/train_models.py --external-csv ...` format demo (aliased column names: `time`, `plot_id`, `moisture`, `temp`, `rain`, ...). It is far too short to train on; it shows the accepted format. |

PowerShell example:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/api/v1/fields `
  -ContentType "application/json" -Body (Get-Content examples\field_create.json -Raw)
```
