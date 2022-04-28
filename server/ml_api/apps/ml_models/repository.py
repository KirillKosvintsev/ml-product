import os
import shutil
from typing import List, Dict, Optional
from datetime import datetime
import pickle
from uuid import UUID

import pandas as pd
from sqlalchemy.orm import Session
from fastapi.responses import FileResponse

from ml_api.common.config import ROOT_DIR
from ml_api.apps.ml_models.models import Model


class BaseCrud:

    def __init__(self, user):
        self.user_id = str(user.id)
        self.user_path = os.path.join(ROOT_DIR, self.user_id, 'models')
        if not os.path.exists(self.user_path):
            os.makedirs(self.user_path)

    def file_path(self, filename):
        return os.path.join(self.user_path, filename + '.pickle')


class ModelPostgreCRUD(BaseCrud):

    def __init__(self, session: Session, user):
        super().__init__(user)
        self.session = session

    # CREATE
    def new_model(self, model_name: str, csv_id: str, task_type: str,
                  composition_type: str, hyperparams: List, metrics: Dict):
        """ DEV USE: Save model info to PostgreDB"""
        new_obj = Model(
            name=model_name,
            filepath=self.file_path(model_name),
            user_id=self.user_id,
            csv_id=csv_id,
            create_date=str(datetime.now()),
            task_type=task_type,
            composition_type=composition_type,
            hyperparams=hyperparams,
            metrics=metrics
        )
        self.session.add(new_obj)
        self.session.commit()

    # READ
    def read_model_by_name(self, model_name: str):
        filepath = self.file_path(model_name)
        model = self.session.query(Model.id, Model.name, Model.create_date, Model.task_type, Model.composition_type,
                                   Model.hyperparams, Model.metrics).filter(Model.filepath == filepath).first()
        if model is not None:
            return model
        return None

    def read_all_models_by_user(self):
        return self.session.query(Model.name, Model.create_date, Model.task_type, Model.composition_type).\
            filter(Model.user_id == self.user_id).all()

    # UPDATE
    def update_model(self, model_name: str, query: Dict):
        filepath = self.file_path(model_name)
        if query.get('name'):
            query['filepath'] = self.file_path(query['name'])
        self.session.query(Model).filter(Model.filepath == filepath).update(query)
        self.session.commit()

    # DELETE
    def delete_model(self, model_name: str):
        filepath = self.file_path(model_name)
        self.session.query(Model).filter(Model.filepath == filepath).delete()
        self.session.commit()


class ModelPickleCRUD(BaseCrud):

    # CREATE/UPDATE
    def save_model(self, model_name: str, model):
        """ DEV USE: Save model in the pickle format"""
        model_path = self.file_path(model_name)
        with open(model_path, 'wb') as handle:
            pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # READ
    def read_model(self, model_name: str) -> object:
        """ DEV USE: Load model from the pickle format"""
        model_path = self.file_path(model_name)
        with open(model_path, 'rb') as handle:
            model = pickle.load(handle)
        return model

    def download_pickled_model(self, model_name: str):
        model_path = self.file_path(model_name)
        return FileResponse(path=model_path, filename=str(model_name + '.pickle'))

    # UPDATE
    def rename_model(self, model_name: str, new_model_name: str):
        old_path = self.file_path(model_name)
        new_path = self.file_path(new_model_name)
        shutil.move(old_path, new_path)

    # DELETE
    def delete_model(self, model_name: str):
        os.remove(self.file_path(model_name))
