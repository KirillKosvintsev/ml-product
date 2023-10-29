import traceback

import pandas as pd

from ml_api.apps.ml_models.models import ModelMetadata
from ml_api.apps.ml_models import utils, errors


class ModelPredictorService:
    def __init__(self, user_id, model_meta: ModelMetadata, model):
        self._user_id = user_id
        self.model = model
        # self.task_type = model_meta.task_type
        self.model_id = model_meta.id
        self.dataframe_id = model_meta.dataframe_id
        self.feature_columns = model_meta.feature_columns
        self.target_column = model_meta.target_column

    async def predict(self, features: pd.DataFrame):
        try:
            predictions = pd.Series(self.model.predict(features), name=self.target_column)
        except Exception as err:
            print(traceback.format_exc())
            error_type = type(err).__name__
            error_description = str(err)
            raise errors.ModelPredictionError(f"{error_type}: {error_description}")
        results_df = utils.get_predictions_df(features, predictions)
        return results_df