from typing import List, Optional

from bunnet import PydanticObjectId
import pandas as pd

from ml_api.apps.dataframes.repositories.repository_manager import \
    DataframeRepositoryManager
from ml_api.apps.dataframes.model import DataFrameMetadata
from ml_api.apps.dataframes import utils, errors
from ml_api.apps.dataframes.services.dataframe_service import DataframeService
from ml_api.apps.dataframes.services.methods_service import DataframeMethodsService


class DataframeServiceFacade:

    def __init__(self, user_id):
        self._user_id = user_id
        self.repository = DataframeRepositoryManager(self._user_id)
        self.dataframe_service = DataframeService(self._user_id)
        self.dataframe_methods_service = DataframeMethodsService(self._user_id)

    def save_predictions_dataframe(
            self, df_filename: str, pred_df: pd.DataFrame) -> DataFrameMetadata:
        meta_created = self.repository.save_prediction_dataframe(
                    pred_df, df_filename)
        return meta_created

    def check_prediction_filename(self, filename: str):
        self.dataframe_service._check_filename_exists(filename)

    def check_dataframe_not_prediction(self, dataframe_id: PydanticObjectId):
        self.dataframe_service._ensure_not_prediction(dataframe_id)

    def delete_prediction(self, prediction_id: PydanticObjectId):
        self.repository.delete_dataframe(prediction_id)

    def get_feature_target_column_names(
            self, dataframe_id: PydanticObjectId) -> (List[str], Optional[str]):
        return self.dataframe_methods_service.get_feature_target_column_names(
            dataframe_id)

    def get_feature_target_df_supervised(
            self, dataframe_id: PydanticObjectId) -> (pd.DataFrame, pd.Series):
        return self.dataframe_methods_service.get_feature_target_df_supervised(
            dataframe_id)

    def get_feature_target_df(
            self, dataframe_id: PydanticObjectId
    ) -> (pd.DataFrame, Optional[pd.Series]):
        return self.dataframe_methods_service.get_feature_target_df(
            dataframe_id)

    def copy_pipeline_for_prediction(self, id_from: PydanticObjectId,
                            id_to: PydanticObjectId):
        df = self.dataframe_methods_service.copy_pipeline_for_prediction(
            id_from, id_to)
        return df
