from unicodedata import numeric
import pandas as pd
import numpy as np
from datetime import datetime
from outliers import smirnov_grubbs as grubbs
from typing import List, Union, Dict, Callable, Tuple, Optional
# from regex import D
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import IsolationForest
from scipy.stats import mode
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.feature_selection import SelectKBest, f_classif, SelectFpr, SelectFwe, SelectFdr, RFE, SelectFromModel
from sklearn.decomposition import PCA
import pickle

from sympy import numer

from ml_api.apps.documents.models import Document
from ml_api.apps.documents.repository import DocumentFileCRUD, DocumentPostgreCRUD


class DocumentService:

    def __init__(self, db, user):
        self._db = db
        self._user = user

    def upload_document_to_db(self, file, filename: str):
        document_name = DocumentPostgreCRUD(self._db, self._user).read_document_column(filename, column=None)
        if document_name is not None:
            return document_name[0]
        DocumentFileCRUD(self._user).upload_document(filename, file)
        DocumentPostgreCRUD(self._db, self._user).new_document(filename)
        return True

    def read_document_from_db(self, filename: str) -> pd.DataFrame:
        document_info = DocumentPostgreCRUD(self._db, self._user).read_document_column(filename, column=None)
        if document_info is None:
            return None
        df = DocumentFileCRUD(self._user).read_document(filename)
        return df

    def download_document_from_db(self, filename: str):
        file = DocumentFileCRUD(self._user).download_document(filename)
        return file

    def rename_document(self, filename: str, new_filename: str):
        DocumentFileCRUD(self._user).rename_document(filename, new_filename)
        query = {
            'name': new_filename
        }
        DocumentPostgreCRUD(self._db, self._user).update_document(filename, query)

    def update_change_date_in_db(self, filename: str):
        query = {
            'change_date': str(datetime.now())
        }
        DocumentPostgreCRUD(self._db, self._user).update_document(filename, query)

    def delete_document_from_db(self, filename: str):
        DocumentFileCRUD(self._user).delete_document(filename)
        DocumentPostgreCRUD(self._db, self._user).delete_document(filename)

    def read_pipeline(self, filename: str):
        pipeline = DocumentPostgreCRUD(self._db, self._user).read_document_column(filename, column='pipeline')
        return pipeline

    def update_pipeline(self, filename: str, method: str):
        pipeline = self.read_pipeline(filename)
        pipeline.append(method)
        query = {
            'pipeline': pipeline
        }
        DocumentPostgreCRUD(self._db, self._user).update_document(filename, query)

    def read_documents_columns(self, filename: str):
        df = DocumentFileCRUD(self._user).read_document(filename)
        return df.columns.to_list()

    def read_column_marks(self, filename: str):
        column_marks = dict(DocumentPostgreCRUD(self._db, self._user).read_document_column(filename,
                                                                                           column='column_marks'))
        return column_marks

    def update_column_marks(self, filename: str, column_marks: Dict[str, Union[List[str], str]]):
        query = {
            'column_marks': column_marks
        }
        DocumentPostgreCRUD(self._db, self._user).update_document(filename, query)

    # DOCUMENT CHANGING METHODS
    def remove_duplicates(self, filename: str):
        document = DocumentFileCRUD(self._user).read_document(filename)
        document = document.drop_duplicates()
        DocumentFileCRUD(self._user).update_document(filename, document)
        self.update_change_date_in_db(filename)

    def drop_na(self, filename: str):
        document = DocumentFileCRUD(self._user).read_document(filename)
        document = document.dropna()
        DocumentFileCRUD(self._user).update_document(filename, document)
        self.update_change_date_in_db(filename)

    def standardize_features(self, filename: str):
        df = DocumentFileCRUD(self._user).read_document(filename)
        numeric_columns = self.read_column_marks(filename)['numeric']
        sc = StandardScaler()
        df[numeric_columns] = pd.DataFrame(sc.fit_transform(df[numeric_columns]), df.index, numeric_columns)
        DocumentFileCRUD(self._user).update_document(filename, df)
        self.update_change_date_in_db(filename)
        self.update_pipeline(filename, method='standardize_features')

    def outliers_IsolationForest(self, filename: str, n_estimators: int, contamination: float):
        document = DocumentFileCRUD(self._user).read_document(filename)
        numeric_columns = self.read_column_marks(filename)['numeric']
        IF = IsolationForest(n_estimators=n_estimators, contamination=contamination)
        document_with_forest = document.join(pd.DataFrame(IF.fit_predict(document[numeric_columns]),
                                                          index=document.index, columns=['isolation_forest']),
                                             how='left')
        document_with_forest = document_with_forest.loc[document_with_forest['isolation_forest'] == 1]
        document = document_with_forest.drop("isolation_forest", axis=1)
        DocumentFileCRUD(self._user).update_document(filename, document)
        self.update_change_date_in_db(filename)

    def outlier_three_sigma(self, filename: str):
        document = DocumentFileCRUD(self._user).read_document(filename)
        numeric_columns = self.read_column_marks(filename)['numeric']
        document[numeric_columns] = document.loc[(document[numeric_columns] - document[numeric_columns].mean()).abs() < 3 * document.std(), numeric_columns].dropna(axis=0, how='any')
        DocumentFileCRUD(self._user).update_document(filename, document)
        self.update_change_date_in_db(filename)

    def miss_insert_mean_mode(self, filename: str):
        document = DocumentFileCRUD(self._user).read_document(filename)
        numeric_columns = self.read_column_marks(filename)['numeric']
        categorical = self.read_column_marks(filename)['categorical']
        for feature in list(document):
            if feature in categorical:
                fill_value = mode(document[feature]).mode[0]
            elif feature in numeric_columns:
                fill_value = document[feature].mean()
            document[feature].fillna(fill_value, inplace=True)

    def outlier_grubbs(self, filename: str, alpha: float):
        document = DocumentFileCRUD(self._user).read_document(filename)
        numeric_columns = self.read_column_marks(filename)['numeric']
        for col in numeric_columns:
            document = document.drop(
                grubbs.two_sided_test_indices(document[col], alpha)
            ).reset_index().drop('index', axis=1)
        DocumentFileCRUD(self._user).update_document(filename, document)
        self.update_change_date_in_db(filename)

    def miss_linear_imputer(self, filename: str):
        document = DocumentFileCRUD(self._user).read_document(filename)
        numeric_columns = self.read_column_marks(filename)['numeric']
        document[numeric_columns] = pd.DataFrame(IterativeImputer().fit_transform(document[numeric_columns]))  # default estimator = BayesianRidge()
        DocumentFileCRUD(self._user).update_document(filename, document)
        self.update_change_date_in_db(filename)

    def outliers_EllipticEnvelope(self, filename: str, contamination: float = 0.1):
        document = DocumentFileCRUD(self._user).read_document(filename)
        numeric_columns = self.read_column_marks(filename)['numeric']
        outliers = EllipticEnvelope(contamination=contamination).fit_predict(document[numeric_columns])
        document[numeric_columns] = document.loc[outliers == 1, numeric_columns].reset_index().drop('index', axis=1)
        DocumentFileCRUD(self._user).update_document(filename, document)
        self.update_change_date_in_db(filename)
    
    def outliers_LocalFactor(
        self,
        filename: str,
        n_neighbors: int = 20,
        algorithm: str = 'auto',
        contamination: Union[str, float] = 'auto'
    ):
        document = DocumentFileCRUD(self._user).read_document(filename)
        numeric_columns = self.read_column_marks(filename)['numeric']
        outliers = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            algorithm=algorithm,
            contamination=contamination
        ).fit_predict(document[numeric_columns])
        document[numeric_columns] = document.loc[outliers == 1, numeric_columns].reset_index().drop('index', axis=1)
        DocumentFileCRUD(self._user).update_document(filename, document)
        self.update_change_date_in_db(filename)

    def outliers_Approximate(self, filename: str, deviation: int):
        document = DocumentFileCRUD(self._user).read_document(filename)
        numeric_columns = self.read_column_marks(filename)['numeric']
        M = document[numeric_columns]
        u, s, vh = np.linalg.svd(M, full_matrices=True)
        Mk_rank = np.linalg.matrix_rank(M) - deviation
        Uk, Sk, VHk = u[:, :Mk_rank], np.diag(s)[:Mk_rank, :Mk_rank], vh[:Mk_rank, :]
        Mk = pd.DataFrame(np.dot(np.dot(Uk, Sk), VHk), index=M.index, columns=M.columns)
        delta = abs(Mk - M)
        document = document.drop(delta.idxmax())
        DocumentFileCRUD(self._user).update_document(filename, document)
        self.update_change_date_in_db(filename)
    
    def fs_select_k_best(
        self,
        filename: str,
        score_func: Callable[
            [np.ndarray, np.ndarray],
            Tuple[np.ndarray, np.ndarray]
        ] = f_classif,
        k: Union[int, str] = 10
    ):
        document = DocumentFileCRUD(self._user).read_document(filename)
        X, y = document.drop('target', axis=1), document['target']
        selector = SelectKBest(score_func=score_func, k=k)
        selector.fit(X, y)
        document = pd.DataFrame(selector.transform(X), columns=document.columns[selector.get_support(indices=True)])
        document['target'] = y
        self.update_change_date_in_db(filename)
        DocumentFileCRUD(self._user).update_document(filename, document)
        self.update_pipeline(filename, method='fs_select_k_best')

    def fs_select_fpr(
        self,
        filename: str,
        score_func: Callable[
            [np.ndarray, np.ndarray],
            Tuple[np.ndarray, np.ndarray]
        ] = f_classif,
        alpha: Union[int, str] = 0.05
    ):
        document = DocumentFileCRUD(self._user).read_document(filename)
        X, y = document.drop('target', axis=1), document['target']
        selector = SelectFpr(score_func=score_func, alpha=alpha)
        selector.fit(X, y)
        document = pd.DataFrame(selector.transform(X), columns=document.columns[selector.get_support(indices=True)])
        document['target'] = y
        self.update_change_date_in_db(filename)
        DocumentFileCRUD(self._user).update_document(filename, document)
        self.update_pipeline(filename, method='fs_select_fpr')

    def fs_select_fwe(
        self,
        filename: str,
        score_func: Callable[
            [np.ndarray, np.ndarray],
            Tuple[np.ndarray, np.ndarray]
        ] = f_classif,
        alpha: Union[int, str] = 0.05
    ):
        document = DocumentFileCRUD(self._user).read_document(filename)
        X, y = document.drop('target', axis=1), document['target']
        selector = SelectFwe(score_func=score_func, alpha=alpha)
        selector.fit(X, y)
        document = pd.DataFrame(selector.transform(X), columns=document.columns[selector.get_support(indices=True)])
        document['target'] = y
        self.update_change_date_in_db(filename)
        DocumentFileCRUD(self._user).update_document(filename, document)
        self.update_pipeline(filename, method='fs_select_fwe')

    def fs_select_fdr(
        self,
        filename: str,
        score_func: Callable[
            [np.ndarray, np.ndarray],
            Tuple[np.ndarray, np.ndarray]
        ] = f_classif,
        alpha: Union[int, str] = 0.05
    ):
        document = DocumentFileCRUD(self._user).read_document(filename)
        X, y = document.drop('target', axis=1), document['target']
        selector = SelectFdr(score_func=score_func, alpha=alpha)
        selector.fit(X, y)
        document = pd.DataFrame(selector.transform(X), columns=document.columns[selector.get_support(indices=True)])
        document['target'] = y
        self.update_change_date_in_db(filename)
        DocumentFileCRUD(self._user).update_document(filename, document)
        self.update_pipeline(filename, method='fs_select_fdr')

    def fs_pca(self, filename: str, n_components: Optional[int]):
        document = DocumentFileCRUD(self._user).read_document(filename)
        X, y = document.drop('target', axis=1), document['target'] 
        if n_components is None:
            pca = PCA().fit(X)
            n_components = len(pca.singular_values_[pca.singular_values_ > 1])
        document = pd.DataFrame(
            PCA(n_components=n_components).fit_transform(X),
            columns=[f'PC{i}' for i in range(1, n_components + 1)]
        )
        document['target'] = y
        self.update_change_date_in_db(filename)
        DocumentFileCRUD(self._user).update_document(filename, document)
        self.update_pipeline(filename, method='fs_pca')
        
          
### ---------------------------------------------UNCHECKED--------------------------------------------------------------

    # OCSVM на выходе 1 - выброс, -1 - не выброс
    # iter - количество итераций, если -1, то нет ограничений
    def outliers_OneClassSVM(self, filename: str, iters: float):
        df = DocumentFileCRUD(self._user).read_document(filename)
        dataset = df.copy()
        OCSVM = OneClassSVM(kernel='rbf', gamma='auto', max_iter=iters)
        df_with_svm = dataset.join(pd.DataFrame(OCSVM.fit_predict(dataset),
                                                index=dataset.index, columns=['svm']), how='left')
        df = df_with_svm.loc[df_with_svm['svm'] != -1]
        DocumentFileCRUD(self._user).update_document(filename, df)
        self.update_change_date_in_db(filename)

    # В обрабатываемом датафрейме значения должны быть только числовыми
    # В старом коде этот метод возвращал не то, что внутри квантилей, а наоборот то, что вне(то, что внутри - удалял)
    def outlier_interquartile_distance(self, filename: str, low_quantile: float, up_quantile: float, coef: float):
        df = DocumentFileCRUD(self._user).read_document(filename)
        quantile = df.quantile([low_quantile, up_quantile])
        for column in df:
            low_lim = quantile[column][low_quantile]
            up_lim = quantile[column][up_quantile]
            df = df.loc[df[column] >= low_lim - coef * (up_lim - low_lim)]. \
                loc[df[column] <= up_lim + coef * (up_lim - low_lim)]
        DocumentFileCRUD(self._user).update_document(filename, df)
        self.update_change_date_in_db(filename)
