"""Model metadata and hot reload."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from ..dependencies import AppState, get_state
from ..schemas import ModelInfo

router = APIRouter(prefix="/api/v1/model", tags=["model"])


def _info(state: AppState) -> ModelInfo:
    model = state.model
    return ModelInfo(
        loaded=model is not None,
        model_path=str(state.settings.model_path),
        info=model.info() if model else None,
        message="ML hybrid engine active" if model else
        "No model loaded - rule-based FAO-56 engine active. Run scripts/train_models.py",
    )


@router.get("", response_model=ModelInfo, summary="Loaded model, threshold and test metrics")
def model_info(state: AppState = Depends(get_state)) -> ModelInfo:
    return _info(state)


@router.post("/reload", response_model=ModelInfo, summary="Reload the model file from disk")
def reload_model(state: AppState = Depends(get_state)) -> ModelInfo:
    state.reload_model()
    return _info(state)
