"""Recommendation engine (ML + transparent FAO-56 rules + fallbacks)."""

from .recommender import FieldConfig, Recommendation, RecommendationEngine, field_config_from_row

__all__ = ["FieldConfig", "Recommendation", "RecommendationEngine", "field_config_from_row"]
