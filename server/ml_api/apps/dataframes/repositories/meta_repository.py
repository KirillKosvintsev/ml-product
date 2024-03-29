from typing import List, Dict

from bunnet import PydanticObjectId
from pymongo.errors import DuplicateKeyError

from ml_api.apps.dataframes import schemas, errors
from ml_api.apps.dataframes.model import DataFrameMetadata


class DataFrameMetaCRUD:
    def __init__(self, user_id: PydanticObjectId):
        self.user_id = user_id

    def get(self, dataframe_id: PydanticObjectId) -> DataFrameMetadata:
        dataframe_meta = DataFrameMetadata.get(dataframe_id).run()
        if dataframe_meta is None:
            raise errors.DataFrameNotFoundError(dataframe_id)
        return dataframe_meta

    def get_active(self) -> List[DataFrameMetadata]:
        dataframe_metas = DataFrameMetadata.find(
            DataFrameMetadata.user_id == self.user_id).find(
            DataFrameMetadata.is_prediction == False).to_list()
        return dataframe_metas

    def get_predictions(self) -> List[DataFrameMetadata]:
        dataframe_metas = DataFrameMetadata.find(
            DataFrameMetadata.user_id == self.user_id).find(
            DataFrameMetadata.is_prediction == True).to_list()
        return dataframe_metas

    def get_by_parent_id(self, parent_id: PydanticObjectId
                               ) -> List[DataFrameMetadata]:
        dataframe_metas = DataFrameMetadata.find(
            DataFrameMetadata.user_id == self.user_id).find(
            DataFrameMetadata.parent_id == parent_id).to_list()
        return dataframe_metas

    def get_by_filename(self, filename: str) -> DataFrameMetadata:
        dataframe_meta = DataFrameMetadata.find_one(
            DataFrameMetadata.filename == filename,
            DataFrameMetadata.user_id == self.user_id).run()
        return dataframe_meta

    def create(
            self,
            filename: str,
            parent_id: PydanticObjectId = None,
            is_prediction: bool = False,
            feature_columns_types: schemas.ColumnTypes = schemas.ColumnTypes(
                numeric=[], categorical=[]),
            target_feature: str = None,
            pipeline: List[schemas.ApplyMethodParams] = (),
            feature_importance_report: schemas.FeatureSelectionSummary = None
    ) -> DataFrameMetadata:
        new_obj = DataFrameMetadata(
            parent_id=parent_id,
            filename=filename,
            user_id=self.user_id,
            is_prediction=is_prediction,
            feature_columns_types=feature_columns_types,
            target_feature=target_feature,
            pipeline=pipeline,
            feature_importance_report=feature_importance_report)
        try:
            new_obj.insert()
        except DuplicateKeyError:
            raise errors.FilenameExistsUserError(filename)
        return new_obj

    def update(self, dataframe_id: PydanticObjectId, query: Dict
                     ) -> DataFrameMetadata:
        dataframe_meta = DataFrameMetadata.get(dataframe_id).run()
        if not dataframe_meta:
            raise errors.DataFrameNotFoundError(dataframe_id)
        dataframe_meta.update(query)
        dataframe_meta_updated = DataFrameMetadata.get(dataframe_id).run()
        return dataframe_meta_updated

    def delete(self,
                     dataframe_id: PydanticObjectId) -> DataFrameMetadata:
        dataframe_meta = DataFrameMetadata.get(dataframe_id).run()
        if not dataframe_meta:
            raise errors.DataFrameNotFoundError(dataframe_id)
        dataframe_meta.delete()
        return dataframe_meta
