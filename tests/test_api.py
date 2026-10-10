"""REST API tests (FastAPI TestClient, temporary database)."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

FIELD = {
    "name": "North plot", "crop_type": "maize", "soil_type": "loam", "irrigation_method": "drip",
    "planting_date": "2026-06-01", "area_m2": 2000, "flow_rate_lpm": 120, "latitude": 18.5,
}


def now_hour() -> datetime:
    return datetime.now().replace(minute=0, second=0, microsecond=0)


def create_field(client, **overrides) -> dict:
    resp = client.post("/api/v1/fields", json={**FIELD, **overrides})
    assert resp.status_code == 201, resp.text
    return resp.json()


def upload_history(client, field_id: int, start_sm: float, end_sm: float, hours: int = 48) -> None:
    end = now_hour()
    readings = []
    for i in range(hours):
        ts = end - timedelta(hours=hours - 1 - i)
        sm = start_sm + (end_sm - start_sm) * i / (hours - 1)
        readings.append({"field_id": field_id, "timestamp": ts.isoformat(), "soil_moisture": round(sm, 2),
                         "temperature_c": 30.0, "humidity_pct": 50.0, "rainfall_mm": 0.0})
    resp = client.post("/api/v1/readings/batch", json={"readings": readings})
    assert resp.status_code == 201, resp.text


def test_root_health_and_docs(client):
    assert client.get("/").json()["docs"] == "/docs"
    health = client.get("/health").json()
    assert health["status"] == "ok" and health["database"] is True and health["model_loaded"] is False
    assert client.get("/docs").status_code == 200
    schema = client.get("/openapi.json").json()
    assert "/api/v1/recommendations" in schema["paths"]
    assert "/api/v1/iot/ingest" in schema["paths"]


def test_reference_endpoints(client):
    assert "maize" in client.get("/api/v1/reference/crops").json()
    assert client.get("/api/v1/reference/soils").json()["loam"]["field_capacity"] == 0.27
    assert client.get("/api/v1/reference/irrigation-methods").json()["drip"] == 0.9


def test_field_crud_and_validation_errors(client):
    field = create_field(client)
    fid = field["id"]
    assert client.get(f"/api/v1/fields/{fid}").json()["name"] == "North plot"
    assert len(client.get("/api/v1/fields").json()) == 1
    patched = client.patch(f"/api/v1/fields/{fid}", json={"area_m2": 2500}).json()
    assert patched["area_m2"] == 2500 and patched["crop_type"] == "maize"
    assert client.post("/api/v1/fields", json={**FIELD, "crop_type": "rice"}).status_code == 422
    assert client.post("/api/v1/fields", json={**FIELD, "area_m2": -1}).status_code == 422
    assert client.get("/api/v1/fields/999").status_code == 404
    assert client.delete(f"/api/v1/fields/{fid}").status_code == 204
    assert client.get(f"/api/v1/fields/{fid}").status_code == 404


def test_reading_ingest_flags_invalid_and_duplicates(client):
    fid = create_field(client)["id"]
    ts = now_hour().isoformat()
    ok = client.post("/api/v1/readings", json={"field_id": fid, "timestamp": ts, "soil_moisture": 22.0,
                                                "temperature_c": 28, "humidity_pct": 60})
    assert ok.status_code == 201 and ok.json()["is_valid"] is True
    dup = client.post("/api/v1/readings", json={"field_id": fid, "timestamp": ts, "soil_moisture": 22.0})
    assert dup.json()["duplicate"] is True
    bad = client.post("/api/v1/readings", json={"field_id": fid, "timestamp": (now_hour() + timedelta(minutes=5)).isoformat(),
                                                 "soil_moisture": 85.0})
    body = bad.json()
    assert body["is_valid"] is False and any(i["code"] == "out_of_range" for i in body["issues"])
    history = client.get(f"/api/v1/fields/{fid}/readings").json()
    assert len(history) == 2 and history[0]["is_valid"] is False
    assert client.post("/api/v1/readings", json={"field_id": 999, "soil_moisture": 20}).status_code == 404
    assert client.post("/api/v1/readings", json={"field_id": fid, "soil_moisture": "wet"}).status_code == 422


def test_batch_upload_counts(client):
    fid = create_field(client)["id"]
    upload_history(client, fid, 24, 23, hours=10)
    assert len(client.get(f"/api/v1/fields/{fid}/readings?limit=5").json()) == 5
    assert client.post("/api/v1/readings/batch", json={"readings": []}).status_code == 422


def test_recommendation_rule_based_dry_and_wet(client):
    dry = create_field(client)["id"]
    upload_history(client, dry, 18.0, 16.5)
    rec = client.post("/api/v1/recommendations", json={"field_id": dry, "forecast_rain_24h_mm": 0})
    assert rec.status_code == 201, rec.text
    body = rec.json()
    assert body["irrigation_needed"] is True and body["decision_source"] == "rule_based"
    assert body["duration_min"] > 0 and body["water_volume_l"] > 0 and body["id"] is not None
    assert body["forecast_source"] == "client"

    wet = create_field(client, name="Wet")["id"]
    upload_history(client, wet, 26.0, 25.5)
    body = client.post("/api/v1/recommendations", json={"field_id": wet, "forecast_rain_24h_mm": 0}).json()
    assert body["irrigation_needed"] is False and body["duration_min"] == 0

    stored = client.get(f"/api/v1/recommendations/{body['id']}").json()
    assert stored["summary"] == body["summary"]
    assert len(client.get(f"/api/v1/fields/{dry}/recommendations").json()) == 1
    assert client.get("/api/v1/recommendations/9999").status_code == 404


def test_recommendation_with_inline_reading_and_no_persist(client):
    fid = create_field(client)["id"]
    upload_history(client, fid, 20.0, 19.0)
    payload = {"field_id": fid, "persist": False, "forecast_rain_24h_mm": 0,
               "reading": {"timestamp": (now_hour() + timedelta(minutes=1)).isoformat(), "soil_moisture": None,
                           "temperature_c": 30, "humidity_pct": 50}}
    body = client.post("/api/v1/recommendations", json=payload).json()
    assert body["id"] is None
    assert body["decision_source"] == "fallback_estimated"  # inline reading had no soil moisture
    assert client.get(f"/api/v1/fields/{fid}/recommendations").json() == []


def test_future_reading_does_not_become_decision_time(client):
    fid = create_field(client)["id"]
    upload_history(client, fid, 18.0, 17.0)
    future = (now_hour() + timedelta(days=2)).isoformat()
    flagged = client.post("/api/v1/readings", json={"field_id": fid, "timestamp": future, "soil_moisture": 20.0}).json()
    assert flagged["is_valid"] is False
    body = client.post("/api/v1/recommendations", json={"field_id": fid, "forecast_rain_24h_mm": 0}).json()
    assert body["decision_time"] < future
    assert body["decision_source"] == "rule_based"


def test_recommendation_without_data_uses_conservative_fallback(client):
    fid = create_field(client)["id"]
    body = client.post("/api/v1/recommendations", json={"field_id": fid}).json()
    assert body["decision_source"] == "fallback_conservative" and body["confidence"] == "low"
    assert body["forecast_source"] == "simulated-offline"


def test_iot_flow_with_device_registration_and_command(client):
    fid = create_field(client)["id"]
    upload_history(client, fid, 18.0, 16.8)
    assert client.post("/api/v1/devices", json={"device_id": "esp32-01", "field_id": fid}).status_code == 201
    resp = client.post("/api/v1/iot/ingest?recommend=true",
                       json={"device_id": "esp32-01", "sm": 16.7, "t": 31.0, "h": 45.0, "r": 0.0, "bat": 3.9,
                             "ts": (now_hour() + timedelta(minutes=2)).isoformat()})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["valid"] is True and body["field_id"] == fid
    assert body["cmd"]["irrigate"] is True and body["cmd"]["duration_min"] > 0
    devices = client.get(f"/api/v1/devices?field_id={fid}").json()
    assert devices[0]["last_seen_at"] is not None
    # unregistered device / invalid payload
    assert client.post("/api/v1/iot/ingest", json={"device_id": "ghost", "sm": 20}).status_code == 404
    assert client.post("/api/v1/iot/ingest", json={"device_id": "esp32-01", "ts": 12}).status_code == 422
    assert client.post("/api/v1/iot/ingest", json={"device_id": "esp32-01", "ts": "yesterday"}).status_code == 422
    # UTC ISO with 'Z' is accepted (converted to field-local time)
    z = client.post("/api/v1/iot/ingest", json={"device_id": "esp32-01", "sm": 16.6, "ts": "2020-01-01T00:00:00Z"})
    assert z.status_code == 200
    # epoch timestamp accepted, no command requested
    epoch = int(datetime.now().timestamp()) - 60
    body = client.post("/api/v1/iot/ingest", json={"device_id": "esp32-01", "sm": 99.0, "ts": epoch}).json()
    assert body["valid"] is False and "out_of_range" in body["issues"] and body["cmd"] is None


def test_iot_api_key_enforced(settings):
    from irrigation_advisor.api.main import create_app

    with TestClient(create_app(replace(settings, iot_api_key="s3cret"))) as client:
        fid = create_field(client)["id"]
        client.post("/api/v1/devices", json={"device_id": "esp32-01", "field_id": fid})
        payload = {"device_id": "esp32-01", "sm": 20.0}
        assert client.post("/api/v1/iot/ingest", json=payload).status_code == 401
        assert client.post("/api/v1/iot/ingest", json=payload, headers={"X-Device-Key": "wrong"}).status_code == 401
        assert client.post("/api/v1/iot/ingest", json=payload, headers={"X-Device-Key": "s3cret"}).status_code == 200


def test_irrigation_events_and_analytics(client):
    fid = create_field(client)["id"]
    upload_history(client, fid, 18.0, 17.0, hours=30)
    rec = client.post("/api/v1/recommendations", json={"field_id": fid, "forecast_rain_24h_mm": 0}).json()
    ev = client.post(f"/api/v1/fields/{fid}/irrigation-events",
                     json={"start_time": now_hour().isoformat(), "duration_min": 30, "recommendation_id": rec["id"]})
    assert ev.status_code == 201 and ev.json()["water_volume_l"] == pytest.approx(30 * 120)
    assert len(client.get(f"/api/v1/fields/{fid}/irrigation-events").json()) == 1
    assert client.post(f"/api/v1/fields/{fid}/irrigation-events",
                       json={"start_time": now_hour().isoformat(), "duration_min": -5}).status_code == 422
    daily = client.get(f"/api/v1/fields/{fid}/analytics/daily?days=3").json()
    assert len(daily) == 3 and daily[-1]["applied_water_l"] == pytest.approx(3600)
    assert sum(d["readings"] for d in daily) == 30
    summary = client.get(f"/api/v1/fields/{fid}/analytics/summary?days=7").json()
    assert summary["applied_water_l"] == pytest.approx(3600) and summary["fixed_schedule_estimate_l"] > 0
    assert client.get("/api/v1/fields/999/analytics/daily").status_code == 404


def test_weather_forecast_endpoint(client):
    fid = create_field(client)["id"]
    body = client.get(f"/api/v1/fields/{fid}/weather/forecast?hours=48").json()
    assert body["hours"] == 48 and len(body["hourly"]) == 48
    assert "SIMULATED" in body["disclaimer"]
    assert client.get(f"/api/v1/fields/{fid}/weather/forecast?hours=0").status_code == 422


def test_calculator(client):
    payload = {"crop_type": "tomato", "soil_type": "sandy_loam", "irrigation_method": "drip", "days_since_planting": 80,
               "soil_moisture_pct": 15.0, "area_m2": 1000, "flow_rate_lpm": 50, "et0_mm_day": 5}
    body = client.post("/api/v1/tools/irrigation-calculator", json=payload).json()
    # Zr 1.0 m: TAW = 120 mm, RAW = 48 mm, Dr = 80 mm -> irrigate 80 mm net, 88.9 mm gross, 88 889 L, 1778 min
    assert body["irrigation_needed"] is True
    assert body["taw_mm"] == pytest.approx(120) and body["depletion_mm"] == pytest.approx(80)
    assert body["water_volume_l"] == pytest.approx(80 / 0.9 * 1000, rel=1e-3)
    wet = client.post("/api/v1/tools/irrigation-calculator", json={**payload, "soil_moisture_pct": 22.5}).json()
    assert wet["irrigation_needed"] is False and wet["water_volume_l"] == 0


def test_model_endpoints_without_and_with_model(client, client_with_model):
    info = client.get("/api/v1/model").json()
    assert info["loaded"] is False and "rule-based" in info["message"]
    assert client.post("/api/v1/model/reload").json()["loaded"] is False
    loaded = client_with_model.get("/api/v1/model").json()
    assert loaded["loaded"] is True and "model_version" in loaded["info"]


def test_recommendation_with_trained_model(client_with_model):
    fid = create_field(client_with_model)["id"]
    upload_history(client_with_model, fid, 18.0, 16.5, hours=72)
    body = client_with_model.post("/api/v1/recommendations", json={"field_id": fid, "forecast_rain_24h_mm": 0}).json()
    assert body["decision_source"] in {"ml_hybrid", "rule_based"}
    assert body["model_version"] is not None
    assert body["irrigation_needed"] is True
