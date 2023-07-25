from datetime import datetime
from typing import List, Optional, Dict

from beanie import Document
from beanie.odm.fields import PydanticObjectId
from pymongo import IndexModel, ASCENDING, HASHED
from pydantic import Field

from ml_api.apps.ml_models.specs import AvailableTaskTypes, \
    AvailableModelTypes, AvailableParamsTypes
from ml_api.apps.ml_models.schemas import ModelParams, ModelStatuses


class ModelMetadata(Document):
    filename: str
    user_id: PydanticObjectId
    dataframe_id: PydanticObjectId
    task_type: AvailableTaskTypes
    model_params: ModelParams
    params_type: AvailableParamsTypes
    feature_columns: Optional[List[str]] = []
    target_column: Optional[str] = None
    test_size: Optional[float] = 0.2
    status: ModelStatuses = ModelStatuses.BUILDING
    metrics_report_id: Optional[PydanticObjectId] = None
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())

    class Settings:
        collection = "model_collection"
        indexes = [
            IndexModel([("user_id", ASCENDING)]),
            IndexModel([("user_id", ASCENDING), (
                "filename", ASCENDING)], unique=True),
            IndexModel([("dataframe_id", HASHED)]),
        ]
