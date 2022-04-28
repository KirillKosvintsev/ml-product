from typing import Dict, Optional, Any, List
from functools import partial

from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split, cross_validate
from fastapi.responses import JSONResponse
from fastapi import status, BackgroundTasks
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score, accuracy_score, roc_curve, auc
from sklearn.ensemble import VotingClassifier, StackingClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK, space_eval
from sklearn.preprocessing import label_binarize
import numpy as np

from ml_api.apps.ml_models.repository import ModelPostgreCRUD, ModelPickleCRUD
from ml_api.apps.ml_models.configs.classification_models_config import DecisionTreeClassifierParameters, \
    CatBoostClassifierParameters, AvailableModels
from ml_api.apps.documents.services import DocumentService
from ml_api.apps.ml_models.configs.classification_searchers_config import CLASSIFICATION_SEARCHERS_CONFIG
from ml_api.apps.ml_models.schemas import ModelWithParams


class ModelService:

    def __init__(self, db, user):
        self._db = db
        self._user = user

    def download_model_from_db(self, model_name: str):
        file = ModelPickleCRUD(self._user).download_pickled_model(model_name)
        return file

    def rename_model(self, model_name: str, new_model_name: str):
        ModelPickleCRUD(self._user).rename_model(model_name, new_model_name)
        query = {
            'name': new_model_name
        }
        ModelPostgreCRUD(self._db, self._user).update_model(model_name, query)

    def read_model_info(self, model_name: str):
        model = ModelPostgreCRUD(self._db, self._user).read_model_by_name(model_name=model_name)
        return model

    def read_models_info(self):
        models = ModelPostgreCRUD(self._db, self._user).read_all_models_by_user()
        return models

    def delete_model_from_db(self, model_name: str):
        ModelPickleCRUD(self._user).delete_model(model_name)
        ModelPostgreCRUD(self._db, self._user).delete_model(model_name)

    def train_model(self, task_type: str, composition_type: str,
                    model_params: List[ModelWithParams], params_type: str, document_name: str,
                    model_name: str, background_tasks: BackgroundTasks, test_size: Optional[float] = 0.2):
        # checks if name is available
        model_info = ModelPostgreCRUD(self._db, self._user).read_model_by_name(model_name)
        if model_info is not None:
            return JSONResponse(status_code=status.HTTP_406_NOT_ACCEPTABLE, content=f"The model name '{model_info['name']}' "
                                                                                    f"is already taken")
        # checks if data exists
        document_info = DocumentService(self._db, self._user).read_document_info(filename=document_name)
        if document_info is None:
            return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content="No such csv document")

        # checks composition settings
        if composition_type == 'none' and len(model_params) > 1:
            return JSONResponse(status_code=status.HTTP_406_NOT_ACCEPTABLE, content="If composition type is 'NONE' "
                                                                                    "there should be one model")

        # load data
        data = DocumentService(self._db, self._user).read_document_from_db(document_name)
        target_column = DocumentService(self._db, self._user).read_column_marks(document_name)['target']
        features = data.drop(target_column, axis=1)
        target = data[target_column]

        # checks if classification sample is wrong
        if task_type == 'classification' and target.nunique() == 1:
            return JSONResponse(status_code=status.HTTP_406_NOT_ACCEPTABLE, content="Only one class label in csv")

        ModelPostgreCRUD(self._db, self._user).new_model(model_name=model_name, csv_id=str(document_info['id']),
                                                         task_type=task_type, composition_type=composition_type,
                                                         hyperparams=[], metrics={})

        background_tasks.add_task(self.validate_training, features, target, task_type, composition_type, model_params,
                                  params_type, model_name, test_size)

        return JSONResponse(status_code=status.HTTP_200_OK, content=f"Training of model '{model_name}' "
                                                                    f"starts at background")

    def validate_training(self, features, target, task_type: str, composition_type: str,
                          model_params: List[ModelWithParams], params_type: str,
                          model_name: str, test_size: Optional[float] = 0.2):
        if params_type == 'auto':
            model_params = AutoParamsSearch(task_type=task_type, model_params=model_params,
                                            features=features, target=target).search_params()

        composition = CompositionConstructor(task_type=task_type, composition_type=composition_type,
                                             models_with_params=model_params).build_composition()

        model, metrics = CompositionValidator(task_type=task_type, composition=composition, model_name=model_name,
                                              features=features, target=target, test_size=test_size).validate_model()

        self.save_trained_model(model_name=model_name, model=model, model_params=model_params, metrics=metrics)

    def save_trained_model(self, model_name: str, model, model_params: List[ModelWithParams], metrics):
        ModelPickleCRUD(self._user).save_model(model_name, model)

        hyperparams = []
        for i, model in enumerate(model_params):
            hyperparams.append({model.type.value: model.params})
        query = {
            'hyperparams': hyperparams,
            'metrics': metrics
        }
        ModelPostgreCRUD(self._db, self._user).update_model(model_name=model_name, query=query)

    def predict_on_model(self, filename: str, model_name: str):
        data = DocumentService(self._db, self._user).read_document_from_db(filename)
        target_column = DocumentService(self._db, self._user).read_column_marks(filename)['target']
        features = data.drop(target_column, axis=1)
        model = ModelPickleCRUD(self._user).read_model(model_name)
        predictions = model.predict(features)
        return list(predictions)


class AutoParamsSearch:

    def __init__(self, task_type: str, model_params: List[ModelWithParams], features, target):
        self.task_type = task_type
        self.model_params = model_params
        self.features = features
        self.target = target

    def search_params(self):
        for i, model_data in enumerate(self.model_params):
            self.model_params[i].params = self.validate_model_params(model_data.type)
        return self.model_params

    def validate_model_params(self, model_type: AvailableModels) -> Dict[str, Any]:
        if self.task_type == 'classification':
            search_space = CLASSIFICATION_SEARCHERS_CONFIG.get(model_type.value)
            if self.target.nunique() == 2:
                best = fmin(
                    fn=partial(self.objective_binary, model_type=model_type),
                    space=search_space,
                    algo=tpe.suggest,
                    max_evals=50,
                    show_progressbar=True
                )
                return space_eval(search_space, best)
            else:
                best = fmin(
                    fn=partial(self.objective_multiclass, model_type=model_type),
                    space=search_space,
                    algo=tpe.suggest,
                    max_evals=50,
                    show_progressbar=True
                )
                return space_eval(search_space, best)
        return {}

    def objective_multiclass(self, params, model_type: AvailableModels):
        """
            Auxiliary function for scoring of checking iterating parameters. Multiclass classification task type.

            :param params: checking parameters;
            :param model_type: AvailableModels type string model name;
            :return: dict(loss, params, status)
        """
        print(params)
        model = ModelConstructor(task_type=self.task_type, model_type=model_type, params=params).model
        skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=1)
        score = cross_val_score(estimator=model, X=self.features, y=self.target,
                                scoring='roc_auc_ovr_weighted', cv=skf, n_jobs=-1, error_score="raise")
        return {'loss': -score.mean(), 'params': params, 'status': STATUS_OK}

    def objective_binary(self, params, model_type: AvailableModels):
        """
            Auxiliary function for scoring of checking iterating parameters. Binary classification task type.

            :param params: checking parameters;
            :param model_type: AvailableModels type string model name;
            :return: dict(loss, params, status)
        """
        model = ModelConstructor(task_type=self.task_type, model_type=model_type, params=params).model
        skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=1)
        score = cross_val_score(estimator=model, X=self.features, y=self.target,
                                scoring='roc_auc', cv=skf, n_jobs=-1, error_score="raise")
        return {'loss': -score.mean(), 'params': params, 'status': STATUS_OK}


    # def objective_regression(self, params, task_type, model_type, features, target):
    #     model = ModelConstructor(task_type=task_type, model_type=model_type, params=params).model
    #     skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=1)
    #     score = cross_val_score(estimator=model, X=features, y=target,
    #                             scoring='roc_auc_ovr_weighted', cv=skf, n_jobs=-1, error_score="raise")
    #
    #     return {'loss': -score.mean(), 'params': params, 'status': STATUS_OK}


class CompositionConstructor:
    """
        Returns sklearn composition model with fit(), predict() methods
    """

    def __init__(self, task_type: str, composition_type: str, models_with_params: List[ModelWithParams]):
        self.task_type = task_type
        self.composition_type = composition_type
        self.models_with_params = models_with_params

    def build_composition(self):
        models = []
        if self.composition_type == 'none':
            for model in self.models_with_params:
                composition = ModelConstructor(task_type=self.task_type, model_type=model.type,
                                               params=model.params).model
            return composition
        for i, model in enumerate(self.models_with_params):
            models.append((str(i) + "_" + model.type.value, ModelConstructor(task_type=self.task_type,
                                                                             model_type=model.type,
                                                                             params=model.params).model))
        if self.composition_type == 'simple_voting':
            composition = VotingClassifier(estimators=models, voting='hard')
            return composition
        elif self.composition_type == 'weighted_voting':
            composition = VotingClassifier(estimators=models, voting='soft')
            return composition
        elif self.composition_type == 'stacking':
            final_estimator = GradientBoostingClassifier()
            composition = StackingClassifier(estimators=models, final_estimator=final_estimator)
            return composition


class ModelConstructor:
    """
        Create sklearn model from hyper-parameters
    """

    def __init__(self, task_type: str, model_type: AvailableModels, params=Dict[str, Any]):
        self.model_type = model_type
        self.params = params
        if task_type == 'classification':
            self.model = self.construct_classification_model()
        # elif task_type == 'regression':
        #     self.model = self.construct_regression_model()

    def construct_classification_model(self):
        if self.model_type.value == 'DecisionTreeClassifier':
            model = self.get_tree_classifier()
            return model
        if self.model_type.value == 'CatBoostClassifier':
            model = self.get_catboost_classifier()
            return model
        return None

    # def construct_regression_model(self):
    #     return None

    def get_tree_classifier(self):
        params = DecisionTreeClassifierParameters(**self.params)
        # print(params.dict())
        model = DecisionTreeClassifier(**params.dict())
        return model

    def get_catboost_classifier(self):
        params = CatBoostClassifierParameters(**self.params)
        # print(params.dict())
        model = CatBoostClassifier(**params.dict(), verbose=100)
        return model


class CompositionValidator:

    def __init__(self, task_type, composition, model_name, features, target, test_size):
        self.task_type = task_type
        self.composition = composition
        self.model_name = model_name
        self.features = features
        self.target = target
        self.test_size = test_size

    def validate_model(self):
        if self.task_type == 'classification':
            if self.target.nunique() == 2:
                return self.process_binary_classification()
            else:
                return self.process_multiclass__classification()

    def process_binary_classification(self):
        report = dict(task_type='binary_classification')
        probabilities_ok = True

        features_train, features_valid, target_train, target_valid = train_test_split(self.features, self.target,
                                                                                      test_size=self.test_size,
                                                                                      stratify=self.target)
        self.composition.fit(features_train, target_train)
        predictions = self.composition.predict(features_valid)
        try:
            probabilities = self.composition.predict_proba(features_valid)[:, 1]
        except AttributeError:
            probabilities_ok = False
        # except Exception:
        #     probabilities = self.composition.decision_function(features_valid)[:, 1]
        report['accuracy'] = accuracy_score(target_valid, predictions)
        report['recall'] = recall_score(target_valid, predictions)
        report['precision'] = precision_score(target_valid, predictions)
        report['f1'] = f1_score(target_valid, predictions)
        if probabilities_ok:
            report['roc_auc'] = roc_auc_score(target_valid, probabilities)
            fpr, tpr, _ = roc_curve(target_valid, probabilities)
            report['fpr'] = list(fpr)
            report['tpr'] = list(tpr)
        return self.composition, report

    def process_multiclass__classification(self):
        report = dict(task_type='multiclass_classification')
        probabilities_ok = True

        features_train, features_valid, target_train, target_valid = train_test_split(self.features, self.target,
                                                                                      test_size=self.test_size,
                                                                                      stratify=self.target)
        self.composition.fit(features_train, target_train)
        predictions = self.composition.predict(features_valid)
        try:
            probabilities = self.composition.predict_proba(features_valid)
        except AttributeError:
            probabilities_ok = False
        # except Exception:
        #     probabilities = self.composition.decision_function(features_valid)

        report['accuracy'] = accuracy_score(target_valid, predictions)
        report['recall'] = recall_score(target_valid, predictions, average='weighted')
        report['precision'] = precision_score(target_valid, predictions, average='weighted')
        report['f1'] = f1_score(target_valid, predictions, average='weighted')

        if probabilities_ok:
            classes = list(self.target.unique())
            target_valid = label_binarize(target_valid, classes=classes)
            n_classes = len(classes)

            report['roc_auc_weighted'] = roc_auc_score(target_valid, probabilities, average='weighted', multi_class='ovr')

            fpr = dict()
            tpr = dict()
            roc_auc = dict()

            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(target_valid[:, i], probabilities[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])

            fpr["micro"], tpr["micro"], _ = roc_curve(target_valid.ravel(), probabilities.ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

            # First aggregate all false positive rates
            all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

            # Then interpolate all ROC curves at this points
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(n_classes):
                mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

            # Finally average it and compute AUC
            mean_tpr /= n_classes

            fpr["macro"] = all_fpr
            tpr["macro"] = mean_tpr
            roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

            report['frp_micro'] = list(fpr["micro"])
            report['trp_micro'] = list(tpr["micro"])
            report['frp_macro'] = list(fpr["macro"])
            report['trp_macro'] = list(tpr["macro"])
            report['roc_auc_micro'] = roc_auc["micro"]
            report['roc_auc_macro'] = roc_auc["macro"]

        return self.composition, report
