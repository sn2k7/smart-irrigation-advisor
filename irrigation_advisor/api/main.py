"""FastAPI application factory.

Run locally::

    uvicorn irrigation_advisor.api.main:app --reload

Interactive documentation is served at ``/docs`` (Swagger UI) and ``/redoc``.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .. import __version__
from ..config import Settings, get_settings
from ..data.weather_provider import SimulatedWeatherProvider
from ..db.database import Database
from ..engine.recommender import RecommendationEngine
from ..exceptions import ConflictError, DataValidationError, IrrigationAdvisorError, NotFoundError
from ..logging_config import configure_logging
from ..ml.model_store import try_load_model
from .dependencies import AppState
from .routers import analytics, events, fields, health, iot, model, readings, recommendations, tools, weather

logger = logging.getLogger(__name__)

DESCRIPTION = """
Offline decision support for small-farm irrigation.

* **Readings / IoT** - store sensor data (with validation and quality flags), including a compact ESP32 endpoint.
* **Recommendations** - irrigate or not, how long, how much water, with transparent reasons.
* **Analytics** - daily history and an *estimated* comparison with a fixed schedule.

Weather is **simulated offline** unless you send your own `forecast_rain_24h_mm`.
Model metrics come from **synthetic data** - see `/api/v1/model`.
"""


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build a configured FastAPI app (tests pass their own settings)."""
    settings = settings or get_settings()
    configure_logging(settings.log_level, settings.log_dir)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        db = Database(settings.database_path)
        db.initialise()
        engine = RecommendationEngine(
            model=try_load_model(settings.model_path),
            weather=SimulatedWeatherProvider(settings.climate_profile),
            stale_hours=settings.stale_reading_hours,
        )
        app.state.sia = AppState(settings=settings, db=db, engine=engine)
        logger.info("Smart Irrigation Advisor %s started (model loaded: %s)", __version__, engine.model is not None)
        yield
        logger.info("Shutting down")

    app = FastAPI(
        title=settings.api_title,
        version=__version__,
        description=DESCRIPTION,
        lifespan=lifespan,
    )
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    @app.middleware("http")
    async def timing(request: Request, call_next):  # type: ignore[no-untyped-def]
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        response.headers["X-Process-Time-ms"] = f"{elapsed_ms:.1f}"
        logger.debug("%s %s -> %s (%.1f ms)", request.method, request.url.path, response.status_code, elapsed_ms)
        return response

    @app.exception_handler(NotFoundError)
    async def not_found(_: Request, exc: NotFoundError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(ConflictError)
    async def conflict(_: Request, exc: ConflictError) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(DataValidationError)
    async def invalid_data(_: Request, exc: DataValidationError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.exception_handler(IrrigationAdvisorError)
    async def domain_error(_: Request, exc: IrrigationAdvisorError) -> JSONResponse:
        logger.warning("Domain error: %s", exc)
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(Exception)
    async def unexpected(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(status_code=500, content={"detail": "Internal server error - see server logs"})

    for router in (health, fields, readings, iot, recommendations, events, analytics, weather, model, tools):
        app.include_router(router.router)
    return app


app = create_app()
