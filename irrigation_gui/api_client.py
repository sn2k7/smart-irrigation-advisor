"""Thin, synchronous HTTP client for the REST API (used from worker threads).

Every method maps to one endpoint and returns parsed JSON. Errors are raised
as :class:`ApiError` with the server's ``detail`` message so the UI can show
something meaningful ("Field 7 not found", validation messages, ...).
"""

from __future__ import annotations

from typing import Any

import httpx

DEFAULT_URL = "http://127.0.0.1:8000"
API = "/api/v1"


class ApiError(Exception):
    """HTTP or connection error with a human-readable message."""

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


def _detail(response: httpx.Response) -> str:
    try:
        detail = response.json().get("detail")
    except ValueError:
        return response.text[:300] or response.reason_phrase
    if isinstance(detail, list):  # FastAPI/Pydantic validation errors
        parts = []
        for err in detail:
            loc = ".".join(str(x) for x in err.get("loc", [])[1:])
            parts.append(f"{loc}: {err.get('msg')}" if loc else str(err.get("msg")))
        return "; ".join(parts)
    return str(detail)


class ApiClient:
    """Client for the Smart Irrigation Advisor API.

    Args:
        base_url: e.g. ``http://127.0.0.1:8000``.
        timeout: seconds per request.
        client: optional pre-built ``httpx.Client`` (tests pass a FastAPI TestClient).
    """

    def __init__(self, base_url: str = DEFAULT_URL, timeout: float = 20.0, client: httpx.Client | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = client or httpx.Client(base_url=self.base_url, timeout=timeout)

    def set_base_url(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self._client.base_url = httpx.URL(self.base_url)

    def close(self) -> None:
        self._client.close()

    # -------------------------------------------------------------- core
    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            response = self._client.request(method, path, **kwargs)
        except httpx.TimeoutException as exc:
            raise ApiError(f"Request timed out ({path})") from exc
        except httpx.HTTPError as exc:
            raise ApiError(f"Cannot reach the API at {self.base_url} - is it running? ({exc})") from exc
        if response.status_code >= 400:
            raise ApiError(_detail(response), response.status_code)
        if response.status_code == 204 or not response.content:
            return None
        return response.json()

    # -------------------------------------------------------------- system
    def health(self) -> dict:
        return self._request("GET", "/health")

    def reference(self) -> dict:
        return {
            "crops": self._request("GET", f"{API}/reference/crops"),
            "soils": self._request("GET", f"{API}/reference/soils"),
            "methods": self._request("GET", f"{API}/reference/irrigation-methods"),
        }

    def model_info(self) -> dict:
        return self._request("GET", f"{API}/model")

    def reload_model(self) -> dict:
        return self._request("POST", f"{API}/model/reload")

    # -------------------------------------------------------------- fields
    def list_fields(self) -> list[dict]:
        return self._request("GET", f"{API}/fields")

    def create_field(self, data: dict) -> dict:
        return self._request("POST", f"{API}/fields", json=data)

    def update_field(self, field_id: int, data: dict) -> dict:
        return self._request("PATCH", f"{API}/fields/{field_id}", json=data)

    def delete_field(self, field_id: int) -> None:
        self._request("DELETE", f"{API}/fields/{field_id}")

    def register_device(self, device_id: str, field_id: int, description: str | None = None) -> dict:
        return self._request("POST", f"{API}/devices",
                             json={"device_id": device_id, "field_id": field_id, "description": description})

    def list_devices(self, field_id: int | None = None) -> list[dict]:
        params = {"field_id": field_id} if field_id is not None else None
        return self._request("GET", f"{API}/devices", params=params)

    # -------------------------------------------------------------- readings
    def list_readings(self, field_id: int, limit: int = 500) -> list[dict]:
        return self._request("GET", f"{API}/fields/{field_id}/readings", params={"limit": limit})

    def add_reading(self, data: dict) -> dict:
        return self._request("POST", f"{API}/readings", json=data)

    def add_readings_batch(self, readings: list[dict]) -> dict:
        return self._request("POST", f"{API}/readings/batch", json={"readings": readings})

    # -------------------------------------------------------------- decisions
    def recommend(self, field_id: int, forecast_rain_24h_mm: float | None = None, persist: bool = True) -> dict:
        body: dict[str, Any] = {"field_id": field_id, "persist": persist}
        if forecast_rain_24h_mm is not None:
            body["forecast_rain_24h_mm"] = forecast_rain_24h_mm
        return self._request("POST", f"{API}/recommendations", json=body)

    def list_recommendations(self, field_id: int, limit: int = 50) -> list[dict]:
        return self._request("GET", f"{API}/fields/{field_id}/recommendations", params={"limit": limit})

    def list_events(self, field_id: int, limit: int = 100) -> list[dict]:
        return self._request("GET", f"{API}/fields/{field_id}/irrigation-events", params={"limit": limit})

    def add_event(self, field_id: int, data: dict) -> dict:
        return self._request("POST", f"{API}/fields/{field_id}/irrigation-events", json=data)

    # -------------------------------------------------------------- analytics / weather / tools
    def daily_analytics(self, field_id: int, days: int = 14) -> list[dict]:
        return self._request("GET", f"{API}/fields/{field_id}/analytics/daily", params={"days": days})

    def summary(self, field_id: int, days: int = 30) -> dict:
        return self._request("GET", f"{API}/fields/{field_id}/analytics/summary", params={"days": days})

    def forecast(self, field_id: int, hours: int = 24) -> dict:
        return self._request("GET", f"{API}/fields/{field_id}/weather/forecast", params={"hours": hours})

    def calculator(self, data: dict) -> dict:
        return self._request("POST", f"{API}/tools/irrigation-calculator", json=data)
