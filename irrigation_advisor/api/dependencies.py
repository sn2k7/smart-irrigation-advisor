"""FastAPI dependencies: shared state, DB connections and IoT authentication."""

from __future__ import annotations

import hmac
import sqlite3
import threading
from dataclasses import dataclass, field
from typing import Iterator

from fastapi import Depends, Header, HTTPException, Request, status

from ..config import Settings
from ..db.database import Database
from ..engine.recommender import RecommendationEngine
from ..ml.model_store import IrrigationModel, try_load_model


@dataclass
class AppState:
    """Objects shared by all requests (attached to ``app.state.sia``)."""

    settings: Settings
    db: Database
    engine: RecommendationEngine
    lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def model(self) -> IrrigationModel | None:
        return self.engine.model

    def reload_model(self) -> IrrigationModel | None:
        """(Re)load the model file; the engine keeps working rule-based if it fails."""
        with self.lock:
            self.engine.model = try_load_model(self.settings.model_path)
            return self.engine.model


def get_state(request: Request) -> AppState:
    return request.app.state.sia


def get_conn(state: AppState = Depends(get_state)) -> Iterator[sqlite3.Connection]:
    """One transaction per request."""
    with state.db.connection() as conn:
        yield conn


def verify_device_key(
    state: AppState = Depends(get_state), x_device_key: str | None = Header(default=None)
) -> None:
    """Require ``X-Device-Key`` when ``SIA_IOT_API_KEY`` is configured."""
    expected = state.settings.iot_api_key
    if not expected:
        return
    if x_device_key is None or not hmac.compare_digest(x_device_key, expected):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing X-Device-Key")
