"""Push the latest Open-Meteo hours into a field as 'virtual sensor' readings (API must run).

Lets the dashboard work for the configured location without hardware:
soil moisture (0-27 cm model layers), temperature, humidity and rain of the last
N hours are posted to POST /api/v1/readings/batch (duplicates are skipped).
Start the API with SIA_WEATHER_PROVIDER=openmeteo so forecasts and ET0 are live too.

    python scripts/openmeteo_live_ingest.py --field-id 1
"""

from __future__ import annotations

import argparse
import math

import _bootstrap  # noqa: F401

from irrigation_advisor.config import get_settings
from irrigation_advisor.data.openmeteo import OpenMeteoWeatherProvider
from irrigation_gui.api_client import DEFAULT_URL, ApiClient


def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--field-id", type=int, required=True)
    parser.add_argument("--api-url", default=DEFAULT_URL)
    parser.add_argument("--hours", type=int, default=72)
    args = parser.parse_args()

    api = ApiClient(args.api_url)
    field = next((f for f in api.list_fields() if f["id"] == args.field_id), None)
    if field is None:
        raise SystemExit(f"Field {args.field_id} not found")
    lon = field["longitude"] if field["longitude"] is not None else settings.default_longitude
    provider = OpenMeteoWeatherProvider(lon, settings.timezone)
    df = provider.recent_readings(field["latitude"], args.hours, longitude=lon)

    def clean(v):
        return None if v is None or (isinstance(v, float) and math.isnan(v)) else round(float(v), 3)

    readings = [{"field_id": field["id"], "timestamp": ts.isoformat(), "soil_moisture": clean(r.soil_moisture),
                 "temperature_c": clean(r.temperature_c), "humidity_pct": clean(r.humidity_pct),
                 "rainfall_mm": clean(r.rainfall_mm) or 0.0, "device_id": "open-meteo"}
                for ts, r in zip(df["timestamp"], df.itertuples(index=False))]
    result = api.add_readings_batch(readings)
    print(f"Posted {result['received']} hours: {result['stored']} stored, {result['duplicates']} duplicates, "
          f"{result['invalid']} flagged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
