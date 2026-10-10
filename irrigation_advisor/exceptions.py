"""Domain-specific exceptions. The API layer maps them to HTTP status codes."""


class IrrigationAdvisorError(Exception):
    """Base class for all project errors."""


class DataValidationError(IrrigationAdvisorError):
    """Input data is structurally invalid (missing columns, wrong types...)."""


class NotFoundError(IrrigationAdvisorError):
    """A requested database entity does not exist."""


class ConflictError(IrrigationAdvisorError):
    """An entity already exists or violates a uniqueness constraint."""


class ModelNotAvailableError(IrrigationAdvisorError):
    """No trained model is loaded (the engine then falls back to rules)."""


class ModelSchemaMismatchError(IrrigationAdvisorError):
    """A saved model was trained with a different feature schema."""
