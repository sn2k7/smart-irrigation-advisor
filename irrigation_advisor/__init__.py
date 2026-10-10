"""Smart Irrigation Advisor for Small Farmers.

A fully offline backend that decides whether a field needs irrigation,
how long to run the pump/valve and how much water that will consume.

Sub-packages
------------
agronomy    FAO-56 based crop/soil parameters and water-balance maths.
data        Synthetic data generation, CSV loading, validation, cleaning
            and feature engineering (shared by training *and* inference).
ml          Model pipelines, leakage-safe splitting, training, evaluation.
engine      Transparent recommendation engine with fallback rules.
simulation  Closed-loop field simulator comparing irrigation strategies.
db          SQLite persistence layer.
api         FastAPI REST application.
"""

__version__ = "1.0.0"
