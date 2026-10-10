"""Pydantic request/response models (rendered in the interactive ``/docs``).

Hard bounds on numeric inputs only reject *nonsense* (wrong types, absurd
magnitudes). Physically implausible but well-formed sensor values (e.g. a
soil moisture of 85 %) are accepted, stored with ``is_valid = false`` and
reported back - detecting faulty sensors is a feature, not an input error.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..agronomy.crops import CROPS, IRRIGATION_METHODS, SOILS

CropType = Enum("CropType", {k: k for k in sorted(CROPS)}, type=str)  # type: ignore[misc]
SoilType = Enum("SoilType", {k: k for k in sorted(SOILS)}, type=str)  # type: ignore[misc]
IrrigationMethodType = Enum("IrrigationMethodType", {k: k for k in sorted(IRRIGATION_METHODS)}, type=str)  # type: ignore[misc]


def _enum_value(value: Any) -> Any:
    return value.value if isinstance(value, Enum) else value


# ---------------------------------------------------------------- fields
class FieldBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=100, examples=["North plot"])
    crop_type: CropType = Field(..., description="Crop grown on the field")
    soil_type: SoilType = Field(..., description="Dominant soil texture")
    irrigation_method: IrrigationMethodType = Field(..., description="Determines application efficiency")
    planting_date: date = Field(..., description="Sowing / transplanting date (drives crop coefficient)")
    area_m2: float = Field(..., gt=0, le=1_000_000, description="Irrigated area in square metres")
    flow_rate_lpm: float = Field(..., gt=0, le=100_000, description="Pump/valve flow rate in litres per minute")
    latitude: float = Field(18.5, ge=-90, le=90)
    longitude: float | None = Field(None, ge=-180, le=180)
    max_irrigation_minutes: float = Field(180, gt=0, le=1440, description="Safety cap per irrigation run")

    def to_db(self) -> dict[str, Any]:
        return {k: _enum_value(v) for k, v in self.model_dump().items()}


class FieldCreate(FieldBase):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "North plot",
                "crop_type": "maize",
                "soil_type": "loam",
                "irrigation_method": "drip",
                "planting_date": "2026-06-15",
                "area_m2": 2000,
                "flow_rate_lpm": 120,
                "latitude": 18.5,
                "longitude": 73.8,
                "max_irrigation_minutes": 180,
            }
        }
    )


class FieldUpdate(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=100)
    crop_type: CropType | None = None
    soil_type: SoilType | None = None
    irrigation_method: IrrigationMethodType | None = None
    planting_date: date | None = None
    area_m2: float | None = Field(None, gt=0, le=1_000_000)
    flow_rate_lpm: float | None = Field(None, gt=0, le=100_000)
    latitude: float | None = Field(None, ge=-90, le=90)
    longitude: float | None = Field(None, ge=-180, le=180)
    max_irrigation_minutes: float | None = Field(None, gt=0, le=1440)

    def to_db(self) -> dict[str, Any]:
        return {k: _enum_value(v) for k, v in self.model_dump(exclude_unset=True).items() if v is not None}


class FieldOut(FieldBase):
    id: int
    created_at: str
    updated_at: str


# ---------------------------------------------------------------- devices
class DeviceIn(BaseModel):
    device_id: str = Field(..., min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.:-]+$", examples=["esp32-node-01"])
    field_id: int = Field(..., ge=1)
    description: str | None = Field(None, max_length=200)


class DeviceOut(BaseModel):
    device_id: str
    field_id: int
    description: str | None
    created_at: str
    last_seen_at: str | None


# ---------------------------------------------------------------- readings
class SensorValues(BaseModel):
    """Sensor values shared by full and inline readings."""

    timestamp: datetime | None = Field(None, description="Field-local time; defaults to server time")
    soil_moisture: float | None = Field(None, ge=-100, le=200, description="Volumetric water content, %")
    temperature_c: float | None = Field(None, ge=-100, le=150)
    humidity_pct: float | None = Field(None, ge=-50, le=200)
    rainfall_mm: float | None = Field(0.0, ge=0, le=1000, description="Rain since the previous reading (mm)")
    battery_v: float | None = Field(None, ge=0, le=30)
    rssi_dbm: int | None = Field(None, ge=-150, le=0)
    device_id: str | None = Field(None, max_length=64)


class SensorReadingIn(SensorValues):
    field_id: int = Field(..., ge=1)
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "field_id": 1,
                "device_id": "esp32-node-01",
                "timestamp": "2026-07-20T07:00:00",
                "soil_moisture": 21.4,
                "temperature_c": 27.8,
                "humidity_pct": 64.0,
                "rainfall_mm": 0.0,
                "battery_v": 3.9,
            }
        }
    )


class ValidationIssueOut(BaseModel):
    code: str
    field: str
    message: str
    severity: Literal["error", "warning"]


class ReadingIngestResult(BaseModel):
    id: int | None
    field_id: int
    timestamp: str
    duplicate: bool
    is_valid: bool
    issues: list[ValidationIssueOut]


class BatchReadingsIn(BaseModel):
    readings: list[SensorReadingIn] = Field(..., min_length=1, max_length=5000)


class BatchIngestResult(BaseModel):
    received: int
    stored: int
    duplicates: int
    invalid: int
    results: list[ReadingIngestResult]


class ReadingOut(BaseModel):
    id: int
    field_id: int
    device_id: str
    timestamp: str
    soil_moisture: float | None
    temperature_c: float | None
    humidity_pct: float | None
    rainfall_mm: float | None
    battery_v: float | None
    rssi_dbm: int | None
    source: str
    is_valid: bool
    quality_flags: list[str]
    received_at: str


# ---------------------------------------------------------------- IoT
class IoTPayload(BaseModel):
    """Compact payload for ESP32/Arduino nodes (short keys save bytes over LoRa/GSM)."""

    device_id: str = Field(..., min_length=1, max_length=64, examples=["esp32-node-01"])
    sm: float | None = Field(None, ge=-100, le=200, description="Soil moisture % VWC")
    t: float | None = Field(None, ge=-100, le=150, description="Air temperature °C")
    h: float | None = Field(None, ge=-50, le=200, description="Relative humidity %")
    r: float | None = Field(0.0, ge=0, le=1000, description="Rain since last message, mm")
    bat: float | None = Field(None, ge=0, le=30, description="Battery voltage")
    rssi: int | None = Field(None, ge=-150, le=0)
    ts: int | str | None = Field(None, description="Unix epoch seconds (UTC) or ISO time; omitted -> server time")

    @field_validator("ts")
    @classmethod
    def _check_ts(cls, v: int | str | None) -> int | str | None:
        if isinstance(v, int) and not 946_684_800 <= v <= 4_102_444_800:  # 2000-01-01 .. 2100-01-01
            raise ValueError("ts must be a Unix timestamp in seconds")
        if isinstance(v, str):
            text = v.strip()
            try:
                datetime.fromisoformat(text[:-1] + "+00:00" if text.endswith(("Z", "z")) else text)
            except ValueError as exc:
                raise ValueError("ts must be ISO-8601 (e.g. 2026-07-01T06:00:00) or Unix seconds") from exc
        return v


class IoTCommand(BaseModel):
    irrigate: bool
    duration_min: float
    next_check_min: int
    recommendation_id: int | None = None


class IoTResponse(BaseModel):
    accepted: bool
    duplicate: bool
    valid: bool
    issues: list[str]
    field_id: int
    cmd: IoTCommand | None = None


# ---------------------------------------------------------------- recommendations
class RecommendationRequest(BaseModel):
    field_id: int = Field(..., ge=1)
    reading: SensorValues | None = Field(None, description="Optional reading to store before deciding")
    forecast_rain_24h_mm: float | None = Field(
        None, ge=0, le=500, description="Your own 24 h rain forecast; omitted -> offline simulated forecast"
    )
    at: datetime | None = Field(None, description="Decision time; defaults to the latest reading or now")
    persist: bool = Field(True, description="Store the recommendation in the database")
    model_config = ConfigDict(
        json_schema_extra={"example": {"field_id": 1, "forecast_rain_24h_mm": 0.0, "persist": True}}
    )


class RecommendationOut(BaseModel):
    id: int | None = None
    field_id: str
    decision_time: str
    irrigation_needed: bool
    decision_source: Literal["ml_hybrid", "rule_based", "fallback_estimated", "fallback_conservative"]
    confidence: Literal["high", "medium", "low"]
    summary: str
    probability: float | None
    net_depth_mm: float
    gross_depth_mm: float
    water_volume_l: float
    duration_min: float
    capped: bool
    soil_moisture_pct: float | None
    soil_moisture_estimated: bool
    depletion_mm: float | None
    raw_mm: float | None
    taw_mm: float | None
    depletion_ratio: float | None
    etc_24h_mm: float | None
    kc: float | None
    growth_stage: str | None
    forecast_rain_24h_mm: float
    forecast_effective_rain_mm: float
    forecast_source: str
    rule_decision: bool | None
    ml_decision: bool | None
    model_version: str | None
    next_check_hours: int
    reasons: list[str]
    warnings: list[str]
    sensor_issues: list[ValidationIssueOut]
    created_at: str | None = None


# ---------------------------------------------------------------- irrigation events
class IrrigationEventIn(BaseModel):
    start_time: datetime
    duration_min: float = Field(..., ge=0, le=1440)
    water_volume_l: float | None = Field(None, ge=0, le=10_000_000, description="Omitted -> duration x flow rate")
    recommendation_id: int | None = Field(None, ge=1)
    source: Literal["manual", "iot_valve", "scheduled"] = "manual"
    notes: str | None = Field(None, max_length=500)


class IrrigationEventOut(BaseModel):
    id: int
    field_id: int
    recommendation_id: int | None
    start_time: str
    duration_min: float
    water_volume_l: float
    source: str
    notes: str | None
    created_at: str


# ---------------------------------------------------------------- analytics / weather / model
class DailyAnalytics(BaseModel):
    date: str
    readings: int
    invalid_readings: int
    avg_soil_moisture: float | None
    min_soil_moisture: float | None
    max_temperature_c: float | None
    avg_humidity_pct: float | None
    rainfall_mm: float | None
    recommendations: int
    irrigate_recommendations: int
    recommended_water_l: float | None
    irrigation_events: int
    applied_water_l: float | None
    pump_minutes: float | None


class FieldSummary(BaseModel):
    field_id: int
    start: str
    end: str
    days: int
    readings: int
    invalid_readings: int
    rainfall_mm: float
    recommendations: int
    irrigate_recommendations: int
    recommended_water_l: float
    applied_water_l: float
    irrigation_events: int
    fixed_schedule_estimate_l: float
    estimated_saving_vs_fixed_pct: float | None
    note: str


class ForecastOut(BaseModel):
    field_id: int
    start: str
    hours: int
    total_rain_mm: float
    rain_probability: float
    temp_max_c: float
    temp_min_c: float
    mean_humidity_pct: float
    source: str
    disclaimer: str
    hourly: list[dict[str, Any]]


class ModelInfo(BaseModel):
    loaded: bool
    model_path: str
    info: dict[str, Any] | None
    message: str


class HealthOut(BaseModel):
    status: str
    version: str
    database: bool
    model_loaded: bool
    model_version: str | None
    weather_source: str
    time: str


# ---------------------------------------------------------------- calculator
class CalculatorRequest(BaseModel):
    """Stateless FAO-56 calculator - useful for demos and field officers."""

    crop_type: CropType
    soil_type: SoilType
    irrigation_method: IrrigationMethodType
    days_since_planting: float = Field(..., ge=0, le=400)
    soil_moisture_pct: float = Field(..., ge=0, le=60)
    area_m2: float = Field(..., gt=0, le=1_000_000)
    flow_rate_lpm: float = Field(..., gt=0, le=100_000)
    et0_mm_day: float = Field(5.0, ge=0, le=15, description="Reference ET; 5 mm/day is a typical warm day")
    forecast_rain_24h_mm: float = Field(0.0, ge=0, le=500)
    max_irrigation_minutes: float | None = Field(None, gt=0, le=1440)
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "crop_type": "tomato",
                "soil_type": "sandy_loam",
                "irrigation_method": "drip",
                "days_since_planting": 60,
                "soil_moisture_pct": 15.5,
                "area_m2": 1000,
                "flow_rate_lpm": 60,
                "et0_mm_day": 5.5,
                "forecast_rain_24h_mm": 0,
            }
        }
    )


class CalculatorResponse(BaseModel):
    irrigation_needed: bool
    growth_stage: str
    kc: float
    root_depth_m: float
    taw_mm: float
    raw_mm: float
    depletion_mm: float
    depletion_ratio: float
    etc_mm_day: float
    effective_rain_mm: float
    projected_depletion_mm: float
    net_depth_mm: float
    gross_depth_mm: float
    water_volume_l: float
    duration_min: float
    capped: bool
    explanation: list[str]
