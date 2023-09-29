from typing import Dict, Any, List, Tuple

from pydantic import BaseModel

from ml_api.apps.ml_models.specs import AvailableModelTypes
from ml_api.apps.training_reports.models import Report


class ModelParams(BaseModel):
    model_type: AvailableModelTypes
    params: Dict[str, Any]


class ModelTrainingResults(BaseModel):
    model: Any = None
    results: List[Tuple[Report, Any]] = []
