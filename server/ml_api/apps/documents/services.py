from datetime import datetime
from typing import List, Union

import pandas as pd
from fastapi.responses import FileResponse
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, SimpleImputer, KNNImputer
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.linear_model import SGDOneClassSVM, LogisticRegression, \
    LinearRegression
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.feature_selection import f_classif, f_regression, RFE, \
    SelectFromModel, SelectPercentile, SelectFdr, \
    SelectFpr, SelectFwe, SelectKBest
from sklearn.decomposition import PCA

from ml_api.apps.documents.repository import DocumentFileCRUD, \
    DocumentPostgreCRUD
from ml_api.apps.documents.schemas import DocumentShortInfo, DocumentFullInfo, \
    TaskType, PipelineElement, ColumnTypes, \
    ColumnDescription, ReadDocumentResponse, DocumentDescription


class DocumentService:

    def __init__(self, db, user):
        self._db = db
        self._user = user

    def _check_if_document_name_exists(self, filename) -> bool:
        if DocumentPostgreCRUD(self._db, self._user).read_by_name(filename):
            return True
        else:
            return False

    def upload_document_to_db(self, file, filename: str) -> bool:
        if self._check_if_document_name_exists(filename):
            return False
        DocumentFileCRUD(self._user).create(filename, file)
        DocumentPostgreCRUD(self._db, self._user).create(filename)
        return True

    def download_document_from_db(self, filename: str) -> FileResponse:
        file = DocumentFileCRUD(self._user).download(filename)
        return file

    def rename_document(self, filename: str, new_filename: str) -> bool:
        if self._check_if_document_name_exists(new_filename):
            return False
        DocumentFileCRUD(self._user).rename(filename, new_filename)
        query = {'name': new_filename}
        DocumentPostgreCRUD(self._db, self._user).update(filename, query)
        return True

    def delete_document_from_db(self, filename: str) -> bool:
        try:
            DocumentPostgreCRUD(self._db, self._user).delete(filename)
            DocumentFileCRUD(self._user).delete(filename)
        except Exception:
            return False
        return True

    def _read_document(self, filename: str) -> pd.DataFrame:
        df = DocumentFileCRUD(self._user).read_df(filename)
        return df

    def read_all_documents_info(self) -> List[DocumentShortInfo]:
        documents = DocumentPostgreCRUD(self._db, self._user).read_all()
        return documents

    def read_document_info(self, filename: str) -> DocumentFullInfo:
        if self._check_if_document_name_exists(filename):
            document = DocumentPostgreCRUD(self._db, self._user).read_by_name(
                filename=filename)
            return document
        return None

    def read_document_columns(self, filename: str) -> List[str]:
        df = self._read_document(filename)
        return df.columns.to_list()

    def read_document_with_pagination(self,
                                      filename: str,
                                      page: int = 1,
                                      rows_on_page: int = 50
                                      ) -> ReadDocumentResponse:
        if self._check_if_document_name_exists(filename):
            df = self._read_document(filename)
            length = len(df)
            pages_count = (length - 1) // rows_on_page + 1
            start_index = (page - 1) * rows_on_page
            stop_index = page * rows_on_page
            if stop_index < length:
                return {
                    'total': pages_count,
                    'records': df.iloc[start_index:stop_index].fillna(
                        "").to_dict('list')
                }
            elif start_index < length:
                return {
                    'total': pages_count,
                    'records': df.iloc[start_index:].fillna("").to_dict(
                        'list')}
            else:
                return {
                    'total': pages_count,
                    'records': pd.DataFrame().fillna("").to_dict('list')}
        return None

    @staticmethod
    def _create_hist_data(df: pd.DataFrame,
                          column_name: str,
                          bins: int
                          ) -> ColumnDescription:
        ints = df[column_name].value_counts(bins=bins).sort_index(
        ).reset_index()
        ints['start'] = ints['index'].apply(lambda x: x.left)
        ints['end'] = ints['index'].apply(lambda x: x.right)
        ints.drop('index', axis=1, inplace=True)
        ints.columns = ['value', 'left', 'right']
        data = list(ints.to_dict('index').values())
        not_null_count = df[column_name].notna().sum()
        data_type = str(df[column_name].dtype)
        return ColumnDescription(name=column_name,
                                 type='numeric',
                                 not_null_count=not_null_count,
                                 data_type=data_type,
                                 data=data)

    @staticmethod
    def _create_counts_data(df: pd.DataFrame,
                            column_name: str) -> ColumnDescription:
        ints = df[column_name].value_counts(normalize=True).reset_index()
        ints.columns = ['name', 'value']
        data = list(ints.to_dict('index').values())
        not_null_count = df[column_name].notna().sum()
        data_type = str(df[column_name].dtype)
        return ColumnDescription(name=column_name,
                                 type='categorical',
                                 not_null_count=not_null_count,
                                 data_type=data_type,
                                 data=data)

    def create_column_stats(self,
                            df: pd.DataFrame,
                            column_types: ColumnTypes,
                            bins: int
                            ) -> List[ColumnDescription]:
        result = []
        for column_name in column_types.numeric:
            result.append(self._create_hist_data(df=df,
                                                 column_name=column_name,
                                                 bins=bins))
        for column_name in column_types.categorical:
            result.append(self._create_counts_data(df=df,
                                                   column_name=column_name))
        target_name = column_types.target
        if target_name:
            task_type = column_types.task_type.value
            if task_type == 'regression':
                result.append(self._create_hist_data(df=df,
                                                     column_name=target_name,
                                                     bins=bins))
            elif task_type == 'classification':
                result.append(self._create_counts_data(df=df,
                    column_name=target_name))
        return result

    def get_document_columns_info(self,
                                  filename: str,
                                  bins: int = 10
                                  ) -> List[ColumnDescription]:
        if DocumentPostgreCRUD(self._db, self._user).read_by_name(
                filename=filename) is not None:
            df = self._read_document(filename)
            column_types = self.read_column_types(filename)
            if column_types:
                return self.create_column_stats(df, column_types, bins)
            else:
                numeric_columns, categorical_columns = self._split_columns(df)
                column_types = ColumnTypes(numeric=numeric_columns,
                                           categorical=categorical_columns)
                return self.create_column_stats(df, column_types, bins)
        return None

    def get_document_stat_description(self,
                                      filename: str
                                      ) -> DocumentDescription:
        if self._check_if_document_name_exists(filename):
            df = self._read_document(filename)
            result = df.describe()
            result.index = [
                "count", "mean", "std", "min", "first_percentile",
                "second_percentile", "third_percentile",
                "max"
            ]
            return result.to_dict('index')
        return None

    @staticmethod
    def _split_columns(df) -> (List[str], List[str]):
        numeric_columns = df.select_dtypes('number').columns.to_list()
        categorical_columns = df.select_dtypes(
            include=['object', 'category']).columns.to_list()
        return numeric_columns, categorical_columns

    def _validate_column_types(self,
                               filename: str,
                               target_column: str,
                               task_type: TaskType
                               ) -> ColumnTypes:
        df = self._read_document(filename)
        df = df.drop(target_column, axis=1)
        df.info()
        numeric_columns, categorical_columns = self._split_columns(df)
        return ColumnTypes(
            numeric=numeric_columns,
            categorical=categorical_columns,
            target=target_column,
            task_type=task_type.value
        )

    def set_column_types(self,
                         filename: str,
                         target_column: str,
                         task_type: str):
        column_types = self._validate_column_types(filename,
                                                   target_column,
                                                   task_type)
        self.update_column_types(filename, column_types)

    def read_column_types(self, filename: str) -> ColumnTypes:
        column_types = DocumentPostgreCRUD(self._db, self._user).read_by_name(
            filename).column_types
        return column_types

    def update_column_types(self, filename: str, column_types: ColumnTypes):
        query = {'column_types': column_types}
        DocumentPostgreCRUD(self._db, self._user).update(filename, query)

    def set_column_as_categorical(self, filename: str, column_name: str):
        column_types = self.read_column_types(filename)
        column_types.numeric.remove(column_name)
        column_types.categorical.append(column_name)
        self.update_column_types(filename, column_types)
        self._update_change_date_in_db(filename)
        return True

    def set_column_as_numeric(self, filename: str, column_name: str):
        df = self._read_document(filename)
        try:
            df[column_name] = pd.to_numeric(df[column_name])
            DocumentFileCRUD(self._user).update(filename, df)
        except ValueError:
            return False
        self._update_change_date_in_db(filename)
        column_types = self.read_column_types(filename)
        column_types.categorical.remove(column_name)
        column_types.numeric.append(column_name)
        self.update_column_types(filename, column_types)
        return True

    def _update_change_date_in_db(self, filename: str):
        query = {'change_date': str(datetime.now())}
        DocumentPostgreCRUD(self._db, self._user).update(filename, query)

    def read_pipeline(self, filename: str) -> List[PipelineElement]:
        pipeline = DocumentPostgreCRUD(self._db, self._user).read_by_name(
            filename).pipeline
        return pipeline

    def update_pipeline(self,
                        filename: str,
                        function_name: str,
                        param: Union[int, float, str] = None):
        pipeline = self.read_pipeline(filename)
        if param:
            pipeline.append(PipelineElement(function_name=function_name,
                                            param=param))
        else:
            pipeline.append(PipelineElement(function_name=function_name))
        query = {'pipeline': pipeline}
        DocumentPostgreCRUD(self._db, self._user).update(filename, query)

    def apply_pipeline_to_csv(self,
                              filename: str,
                              pipeline: List[PipelineElement]):  
        # to do: find errors
        for function in pipeline:
            if function.param:
                self.apply_function(filename=filename,
                                    function_name=function.function_name,
                                    param=function.param)
            else:
                self.apply_function(filename=filename,
                                    function_name=function.function_name)

    def copy_pipeline(self, filename_orig: str, filename_new: str):
        # to do: add return with errors
        pipeline = self.read_pipeline(filename_orig)
        self.apply_pipeline_to_csv(filename_new, pipeline)

    def apply_function(self,
                       filename: str,
                       function_name: str,
                       param: Union[int, float, str] = None
                       ) -> Union[List, bool]:
        document = self._read_document(filename)
        column_types = self.read_column_types(filename)

        document_operator = DocumentOperator(document, column_types)
        document_operator.apply_function(function_name=function_name,
                                         param=param)

        errors = document_operator.get_errors()
        if len(errors) > 0:
            return False
        if document_operator.is_pipelined():
            self.update_pipeline(filename, function_name=function_name,
                param=param)
        self.update_column_types(filename,
            column_types=document_operator.get_column_types())
        DocumentFileCRUD(self._user).update(filename,
            document_operator.get_df())
        self._update_change_date_in_db(filename)
        return True


class DocumentOperator:

    def __init__(self, document: pd.DataFrame, column_types: ColumnTypes):
        self.df = document
        self.column_types = column_types
        self.update_pipeline = False
        self.error = []
        self.methods_list = {
            'remove_duplicates': self.remove_duplicates,
            'drop_na': self.drop_na,
            'drop_column': self.drop_column,
            'miss_insert_mean_mode': self.miss_insert_mean_mode,
            'miss_linear_imputer': self.miss_linear_imputer,
            'miss_knn_imputer': self.miss_knn_imputer,
            'standardize_features': self.standardize_features,
            'ordinal_encoding': self.ordinal_encoding,
            'one_hot_encoding': self.one_hot_encoding,
            'outliers_isolation_forest': self.outliers_isolation_forest,
            'outliers_elliptic_envelope': self.outliers_elliptic_envelope,
            'outliers_local_factor': self.outliers_local_factor,
            'outliers_one_class_svm': self.outliers_one_class_svm,
            'outliers_sgd_one_class_svm': self.outliers_sgd_one_class_svm,
            'fs_select_percentile': self.fs_select_percentile,
            'fs_select_k_best': self.fs_select_k_best,
            'fs_select_fpr': self.fs_select_fpr,
            'fs_select_fdr': self.fs_select_fdr,
            'fs_select_fwe': self.fs_select_fwe,
            'fs_select_rfe': self.fs_select_rfe,
            'fs_select_from_model': self.fs_select_from_model,
            'fs_select_pca': self.fs_select_pca
        }

    def get_df(self) -> pd.DataFrame:
        return self.df

    def get_column_types(self) -> ColumnTypes:
        return self.column_types

    def get_errors(self) -> List[str]:
        return self.error

    def is_pipelined(self) -> bool:
        return self.update_pipeline

    def no_func_error(self, param=None):
        del param  # func is empty
        self.error.append('no_func')

    def apply_function(self, function_name: str,
                       param: Union[int, float, str] = None):
        func = self.methods_list.get(function_name, self.no_func_error)
        if param:
            func(param=param)
        else:
            func()

    def check_nans(self, columns) -> bool:
        if self.df[columns].isna().sum().sum() > 0:
            return True
        return False

    def check_categorical(self) -> bool:
        if len(self.column_types.categorical) > 0:
            return True
        return False

    # CHAPTER 1: MISSING DATA AND DUPLICATES-----------------------------------

    def remove_duplicates(self):
        self.df.drop_duplicates(inplace=True)

    def drop_na(self):
        self.df.dropna(inplace=True)

    def drop_column(self, param: str):
        column = param
        self.df.drop(column, axis=1, inplace=True)
        self.update_pipeline = True
        if self.column_types:
            try:
                self.column_types.numeric.remove(column)
            except ValueError:
                try:
                    self.column_types.categorical.remove(column)
                except ValueError:
                    pass

    def miss_insert_mean_mode(self):
        numeric_columns = self.column_types.numeric
        categorical_columns = self.column_types.categorical
        self.df[numeric_columns] = pd.DataFrame(
            SimpleImputer(strategy='mean').fit_transform(
                self.df[numeric_columns]),
            self.df.index, numeric_columns)
        self.df[categorical_columns] = pd.DataFrame(
            SimpleImputer(strategy='most_frequent').fit_transform(
                self.df[categorical_columns]), self.df.index,
            categorical_columns)

    def miss_linear_imputer(self):
        numeric_columns = self.column_types.numeric
        self.df[numeric_columns] = pd.DataFrame(
            IterativeImputer().fit_transform(self.df[numeric_columns]),
            self.df.index, numeric_columns)

    def miss_knn_imputer(self):
        numeric_columns = self.column_types.numeric
        self.df[numeric_columns] = pd.DataFrame(
            KNNImputer().fit_transform(self.df[numeric_columns]),
            self.df.index, numeric_columns)

    # CHAPTER 2: FEATURE TRANSFORMATION----------------------------------------
    def standardize_features(self):
        numeric_columns = self.column_types.numeric
        if self.check_nans(numeric_columns):
            self.miss_insert_mean_mode()
        sc = StandardScaler()
        self.df[numeric_columns] = pd.DataFrame(
            sc.fit_transform(self.df[numeric_columns]), self.df.index,
            numeric_columns)
        self.update_pipeline = True

    def ordinal_encoding(self):  # ???? ???????????????? ???? ?????????????????? ?? ????????????
        categorical = self.column_types.categorical
        if self.check_nans(categorical):
            self.miss_insert_mean_mode()
        self.df[categorical] = OrdinalEncoder().fit_transform(
            self.df[categorical])

        self.column_types.numeric.extend(categorical)
        self.column_types.categorical = []
        self.update_pipeline = True

    def one_hot_encoding(self):
        categorical = self.column_types.categorical
        if self.check_nans(categorical):
            self.miss_insert_mean_mode()
        enc = OneHotEncoder()
        enc.fit(self.df[categorical])
        new_cols = enc.get_feature_names(categorical)
        self.df[new_cols] = enc.transform(self.df[categorical]).toarray()
        self.df.drop(categorical, axis=1, inplace=True)

        self.column_types.numeric.extend(new_cols)
        self.column_types.categorical = []
        self.update_pipeline = True

    # CHAPTER 3: OUTLIERS------------------------------------------------------
    def outliers_isolation_forest(self):
        numeric_columns = self.column_types.numeric
        if self.check_nans(numeric_columns):
            self.miss_insert_mean_mode()
        outliers = IsolationForest().fit_predict(self.df[numeric_columns])
        self.df = self.df.loc[outliers == 1].reset_index(drop=True)

    def outliers_elliptic_envelope(self):
        numeric_columns = self.column_types.numeric
        if self.check_nans(numeric_columns):
            self.miss_insert_mean_mode()
        outliers = EllipticEnvelope().fit_predict(self.df[numeric_columns])
        self.df = self.df.loc[outliers == 1].reset_index(drop=True)

    def outliers_local_factor(self):
        numeric_columns = self.column_types.numeric
        if self.check_nans(numeric_columns):
            self.miss_insert_mean_mode()
        outliers = LocalOutlierFactor().fit_predict(self.df[numeric_columns])
        self.df = self.df.loc[outliers == 1].reset_index(drop=True)

    def outliers_one_class_svm(self):
        numeric_columns = self.column_types.numeric
        if self.check_nans(numeric_columns):
            self.miss_insert_mean_mode()
        outliers = OneClassSVM().fit_predict(self.df[numeric_columns])
        self.df = self.df.loc[outliers == 1].reset_index(drop=True)

    def outliers_sgd_one_class_svm(self):
        numeric_columns = self.column_types.numeric
        if self.check_nans(numeric_columns):
            self.miss_insert_mean_mode()
        outliers = SGDOneClassSVM().fit_predict(self.df[numeric_columns])
        self.df = self.df.loc[outliers == 1].reset_index(drop=True)

    # CANCELLED FOR FIRST ITERATION:
    # def outliers_three_sigma(self):
    # def outliers_grubbs(self):
    # def outliers_approximate(self):
    # def outliers_interquartile_distance(self):

    # CHAPTER 4: FEATURE SELECTION (required: all columns are numeric)---------

    def fs_select_percentile(self, param: int = 10):
        percentile = int(param)
        if self.check_categorical():
            self.error.append('categorical_select')
            return
        target = self.column_types.target
        x, y = self.df.drop(target, axis=1), self.df[target]
        if self.column_types.task_type.value == 'classification':
            selector = SelectPercentile(f_classif, percentile=percentile)
        else:
            selector = SelectPercentile(f_regression, percentile=percentile)
        selector.fit(x, y)
        selected_columns = self.df.columns[
            selector.get_support(indices=True)].to_list()
        self.df = pd.DataFrame(selector.transform(x), columns=selected_columns)
        self.df[target] = y
        self.column_types.numeric = selected_columns
        self.update_pipeline = True

    def fs_select_k_best(self, param: int = 10):
        k = int(param)
        if self.check_categorical():
            self.error.append('categorical_select')
            return
        target = self.column_types.target
        x, y = self.df.drop(target, axis=1), self.df[target]
        if self.column_types.task_type.value == 'classification':
            selector = SelectKBest(f_classif, k=k)
        else:
            selector = SelectKBest(f_regression, k=k)
        selector.fit(x, y)
        selected_columns = self.df.columns[
            selector.get_support(indices=True)].to_list()
        self.df = pd.DataFrame(selector.transform(x), columns=selected_columns)
        self.df[target] = y
        self.column_types.numeric = selected_columns
        self.update_pipeline = True

    def fs_select_fpr(self, param: float = 5e-2):
        alpha = param
        if self.check_categorical():
            self.error.append('categorical_select')
            return
        target = self.column_types.target
        x, y = self.df.drop(target, axis=1), self.df[target]
        if self.column_types.task_type.value == 'classification':
            selector = SelectFpr(f_classif, alpha=alpha)
        else:
            selector = SelectFpr(f_regression, alpha=alpha)
        selector.fit(x, y)
        selected_columns = self.df.columns[
            selector.get_support(indices=True)].to_list()
        self.df = pd.DataFrame(selector.transform(x), columns=selected_columns)
        self.df[target] = y
        self.column_types.numeric = selected_columns
        self.update_pipeline = True

    def fs_select_fdr(self, param: float = 5e-2):
        alpha = param
        if self.check_categorical():
            self.error.append('categorical_select')
            return
        target = self.column_types.target
        x, y = self.df.drop(target, axis=1), self.df[target]
        if self.column_types.task_type.value == 'classification':
            selector = SelectFdr(f_classif, alpha=alpha)
        else:
            selector = SelectFdr(f_regression, alpha=alpha)
        selector.fit(x, y)
        selected_columns = self.df.columns[
            selector.get_support(indices=True)].to_list()
        self.df = pd.DataFrame(selector.transform(x), columns=selected_columns)
        self.df[target] = y
        self.column_types.numeric = selected_columns
        self.update_pipeline = True

    def fs_select_fwe(self, param: float = 5e-2):
        alpha = param
        if self.check_categorical():
            self.error.append('categorical_select')
            return
        target = self.column_types.target
        x, y = self.df.drop(target, axis=1), self.df[target]
        if self.column_types.task_type.value == 'classification':
            selector = SelectFwe(f_classif, alpha=alpha)
        else:
            selector = SelectFwe(f_regression, alpha=alpha)
        selector.fit(x, y)
        selected_columns = self.df.columns[
            selector.get_support(indices=True)].to_list()
        self.df = pd.DataFrame(selector.transform(x), columns=selected_columns)
        self.df[target] = y
        self.column_types.numeric = selected_columns
        self.update_pipeline = True

    def fs_select_rfe(self, param: int = None):
        n_features_to_select = int(param)
        if self.check_categorical():
            self.error.append('categorical_select')
            return
        target = self.column_types.target
        x, y = self.df.drop(target, axis=1), self.df[target]
        if self.column_types.task_type.value == 'classification':
            selector = RFE(LogisticRegression(),
                           n_features_to_select=n_features_to_select)
        else:
            selector = RFE(LinearRegression(),
                           n_features_to_select=n_features_to_select)
        selector.fit(x, y)
        selected_columns = self.df.columns[
            selector.get_support(indices=True)].to_list()
        self.df = pd.DataFrame(selector.transform(x), columns=selected_columns)
        self.df[target] = y
        self.column_types.numeric = selected_columns
        self.update_pipeline = True

    def fs_select_from_model(self, param: int = None):
        max_features = int(param)
        if self.check_categorical():
            self.error.append('categorical_select')
            return
        target = self.column_types.target
        x, y = self.df.drop(target, axis=1), self.df[target]
        if self.column_types.task_type.value == 'classification':
            selector = SelectFromModel(LogisticRegression(),
                                       max_features=max_features)
        else:
            selector = SelectFromModel(LinearRegression(),
                                       max_features=max_features)
        selector.fit(x, y)
        selected_columns = self.df.columns[
            selector.get_support(indices=True)].to_list()
        self.df = pd.DataFrame(selector.transform(x), columns=selected_columns)
        self.df[target] = y
        self.column_types.numeric = selected_columns
        self.update_pipeline = True

    def fs_select_pca(self, param: int = None):
        n_components = int(param)
        if self.check_categorical():
            self.error.append('categorical_select')
            return
        target = self.column_types.target
        x, y = self.df.drop(target, axis=1), self.df[target]
        if n_components is None:
            selector = PCA().fit(x)
            n_components = len(
                selector.singular_values_[selector.singular_values_ > 1])
        selector = PCA(n_components=n_components)
        selector.fit(x)
        selected_columns = [f'PC{i}' for i in range(1, n_components + 1)]
        self.df = pd.DataFrame(selector.transform(x), columns=selected_columns)
        self.df[target] = y
        self.column_types.numeric = selected_columns
        self.update_pipeline = True
