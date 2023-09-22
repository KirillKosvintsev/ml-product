from typing import List, Optional

from beanie import PydanticObjectId
import pandas as pd

from ml_api.apps.dataframes.services.file_manager import DataframeFileManagerService
from ml_api.apps.dataframes.services.metadata_manager import DataframeMetadataManagerService
from ml_api.apps.dataframes.models import ColumnTypes, DataFrameMetadata
from ml_api.apps.dataframes import utils, specs, models
from ml_api.apps.dataframes import errors


class DataframeManagerService:
    """
    Работает с pd.Dataframe и отвечает за обработку и изменение данных внутри pandas-датафреймов.
    """
    def __init__(self, user_id):
        self._user_id = user_id
        self.metadata_service = DataframeMetadataManagerService(self._user_id)
        self.file_service = DataframeFileManagerService(self._user_id)

    async def upload_new_dataframe(self, file, filename: str) -> DataFrameMetadata:
        dataframe_meta = await self.file_service.upload_file(file, filename)
        column_types = await self._define_initial_column_types(dataframe_meta.id)
        return await self.metadata_service.set_column_types(dataframe_meta.id,
                                                            column_types)

    async def save_transformed_dataframe(
            self,
            changed_df_meta: models.DataFrameMetadata,
            new_df: pd.DataFrame, method_name: str = 'modified'
    ) -> DataFrameMetadata:
        changed_df_meta.parent_id = changed_df_meta.id
        changed_df_meta.filename = f"{changed_df_meta.filename}_{method_name}"
        return await self._save_dataframe(changed_df_meta, new_df)

    async def save_predictions_dataframe(self, dataframe_id: PydanticObjectId,
                                         df_filename: str,
                                         pred_df: pd.DataFrame,
                                         ) -> DataFrameMetadata:
        original_df_meta = await self.metadata_service.get_dataframe_meta(dataframe_id)
        original_df_meta.parent_id = None
        original_df_meta.filename = df_filename
        original_df_meta.is_prediction = True
        original_df_meta.feature_importance_report = None
        return await self._save_dataframe(original_df_meta, pred_df)

    async def _save_dataframe(self, original_df_meta, pred_df) -> DataFrameMetadata:
        try:
            meta_created = await self.file_service.create_file(pred_df,
                                                               original_df_meta)
        except errors.FilenameExistsUserError:
            original_df_meta.filename = f"{original_df_meta.filename}_{utils.get_random_number()}"
            meta_created = await self.file_service.create_file(pred_df,
                                                               original_df_meta)
        return meta_created

    async def move_dataframe_to_root(self, dataframe_id: PydanticObjectId
                                     ) -> DataFrameMetadata:
        dataframe_meta = await self.metadata_service.get_dataframe_meta(dataframe_id)
        if dataframe_meta.parent_id is not None:
            await self.metadata_service.set_parent_id(dataframe_id, None)
        if dataframe_meta.is_prediction:
            await self.metadata_service.set_is_prediction(dataframe_id, False)
        return await self.metadata_service.get_dataframe_meta(dataframe_id)

    async def _define_initial_column_types(self, dataframe_id: PydanticObjectId
                                           ) -> ColumnTypes:
        df = await self.file_service.read_df_from_file(dataframe_id)
        df = df.convert_dtypes()
        numeric_columns = df.select_dtypes(
            include=["integer", "floating"]).columns.to_list()
        categorical_columns = df.select_dtypes(
            include=["string", "boolean", "category"]).columns.to_list()
        column_types = models.ColumnTypes(
            numeric=numeric_columns, categorical=categorical_columns)
        await self._check_columns_consistency(
            df, column_types.numeric + column_types.categorical)
        await self.file_service.write_df_to_file(dataframe_id, df)
        return column_types

    async def convert_column_to_new_type(self,
             dataframe_id: PydanticObjectId,
             column_name: str, new_type: specs.ColumnType) -> DataFrameMetadata:
        column_types = await self.metadata_service.get_column_types(dataframe_id)
        new_type = new_type.value

        current_type = "categorical" if new_type == "numeric" else "numeric"
        if column_name not in getattr(column_types, current_type):
            raise errors.ColumnNotFoundMetadataError(column_name, current_type)

        getattr(column_types, current_type).remove(column_name)
        getattr(column_types, new_type).append(column_name)

        df = await self.file_service.read_df_from_file(dataframe_id)
        await self._check_columns_consistency(df,
            column_types.numeric + column_types.categorical)
        try:
            if new_type == "categorical":
                converted_column = utils._convert_column_to_categorical(
                    df[column_name])
                if converted_column.nunique() > 1000:
                    raise errors.ColumnCantBeParsedError(column_name,
                        "categorical", "too many unique values")
            else:  # new_type == "numeric"
                converted_column = utils._convert_column_to_numeric(
                    df[column_name])
        except ValueError:
            raise errors.ColumnCantBeParsedError(column_name, new_type,
                                                 "invalid values")
        df[column_name] = converted_column
        df = df.convert_dtypes()
        await self.file_service.write_df_to_file(dataframe_id, df)
        return await self.metadata_service.set_column_types(dataframe_id,
                                                            column_types)

    async def get_feature_target_column_names(self,
            dataframe_id: PydanticObjectId) -> (List[str], Optional[str]):
        """Returns list of feature columns and target column name.
        If target column is not set, returns None instead of target column name.
        If feature columns contain categorical column, raises ColumnNotNumericError."""
        dataframe_meta = await self.metadata_service.get_dataframe_meta(dataframe_id)
        column_types = dataframe_meta.feature_columns_types
        target_column = dataframe_meta.target_feature
        if target_column is not None:
            if target_column in column_types.numeric:
                column_types.numeric.remove(target_column)
            elif target_column in column_types.categorical:
                column_types.categorical.remove(target_column)
            else:
                raise errors.TargetNotFoundError(dataframe_id)
        if len(column_types.categorical) > 0:
            raise errors.ColumnNotNumericError(column_types.categorical[0])
        return column_types.numeric, target_column

    async def get_feature_target_df_supervised(self,
            dataframe_id: PydanticObjectId) -> (pd.DataFrame, pd.Series):
        features, target = await self.get_feature_target_df(dataframe_id)
        if target is None:
            raise errors.TargetNotFoundError(dataframe_id)
        return features, target

    async def get_feature_target_df(self, dataframe_id: PydanticObjectId
                                    ) -> (pd.DataFrame, Optional[pd.Series]):
        feature_columns, target_column = await self.get_feature_target_column_names(
            dataframe_id=dataframe_id)
        df = await self.file_service.read_df_from_file(dataframe_id)
        if target_column is not None:
            df_columns_list = feature_columns + [target_column]
            await self._check_columns_consistency(df, df_columns_list)
            features = df.drop(target_column, axis=1)
            target = df[target_column]
            return features, target
        else:
            await self._check_columns_consistency(df, feature_columns)
            return df, None

    async def _check_columns_consistency(self, df, df_columns_list) -> None:
        if sorted(df.columns.tolist()) != sorted(df_columns_list):
            raise errors.ColumnsNotEqualError()
