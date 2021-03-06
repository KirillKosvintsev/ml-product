import os
import shutil
from typing import List, Dict
from datetime import datetime
import pickle

from sqlalchemy.orm import Session
from fastapi.responses import FileResponse
from ml_api.common.config import ROOT_DIR
from ml_api.apps.ml_models.models import Model
from ml_api.apps.ml_models.schemas import CompositionFullInfo, \
    CompositionShortInfo


class BaseCRUD:

    def __init__(self, user):
        self.user_id = str(user.id)
        self.user_path = os.path.join(ROOT_DIR, self.user_id, 'models')
        if not os.path.exists(self.user_path):
            os.makedirs(self.user_path)

    def file_path(self, filename):
        return os.path.join(self.user_path, filename + '.pickle')


class ModelPostgreCRUD(BaseCRUD):

    def __init__(self, session: Session, user):
        super().__init__(user)
        self.session = session

    # CREATE
    def create(self, model_name: str, csv_id: str, task_type: str, target: str,
               features: List, composition_type: str, composition_params: List,
               stage: str, report: Dict):
        """ DEV USE: Save model info to PostgreDB"""
        new_obj = Model(
            name=model_name,
            filepath=self.file_path(model_name),
            user_id=self.user_id,
            csv_id=csv_id,
            features=features,
            target=target,
            create_date=str(datetime.now()),
            task_type=task_type,
            composition_type=composition_type,
            composition_params=composition_params,
            stage=stage,
            report=report
        )
        self.session.add(new_obj)
        self.session.commit()

    # READ
    def read_by_name(self, model_name: str) -> CompositionFullInfo:
        filepath = self.file_path(model_name)
        model = self.session.query(Model.id, Model.name, Model.csv_id,
            Model.features, Model.target, Model.create_date, Model.task_type,
            Model.composition_type, Model.composition_params, Model.stage,
            Model.report).filter(Model.filepath == filepath).first()
        if model:
            return CompositionFullInfo.from_orm(model)
        return None

    def read_all(self) -> List[CompositionShortInfo]:
        models = self.session.query(Model.name, Model.csv_id, Model.features,
            Model.target, Model.create_date, 
            Model.task_type, Model.composition_type, Model.stage).filter(
            Model.user_id == self.user_id).all()
        result = []
        for model in models:
            result.append(CompositionShortInfo.from_orm(model))
        return result

    # UPDATE
    def update(self, model_name: str, query: Dict):
        filepath = self.file_path(model_name)
        if query.get('name'):
            query['filepath'] = self.file_path(query['name'])
        self.session.query(Model).filter(Model.filepath == filepath).update(
            query)
        self.session.commit()

    # DELETE
    def delete(self, model_name: str):
        filepath = self.file_path(model_name)
        self.session.query(Model).filter(Model.filepath == filepath).delete()
        self.session.commit()


class ModelPickleCRUD(BaseCRUD):

    # CREATE/UPDATE
    def save(self, model_name: str, model):
        """ DEV USE: Save model in the pickle format"""
        model_path = self.file_path(model_name)
        with open(model_path, 'wb') as handle:
            pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # READ
    def load(self, model_name: str):
        """ DEV USE: Load model from the pickle format"""
        model_path = self.file_path(model_name)
        with open(model_path, 'rb') as handle:
            model = pickle.load(handle)
        return model

    def download_pickled(self, model_name: str):
        model_path = self.file_path(model_name)
        return FileResponse(path=model_path,
                            filename=str(model_name + '.pickle'))

    # UPDATE
    def rename(self, model_name: str, new_model_name: str):
        old_path = self.file_path(model_name)
        new_path = self.file_path(new_model_name)
        shutil.move(old_path, new_path)

    # DELETE
    def delete(self, model_name: str):
        os.remove(self.file_path(model_name))
