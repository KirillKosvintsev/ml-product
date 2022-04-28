import os
import shutil
from typing import List, Dict, Optional
from datetime import datetime
from uuid import UUID
import pickle

import pandas as pd
from sqlalchemy.orm import Session
from fastapi.responses import FileResponse
from ml_api.common.config import ROOT_DIR
from ml_api.apps.documents.models import Document


class BaseCrud:

    def __init__(self, user):
        self.user_id = str(user.id)
        self.user_path = os.path.join(ROOT_DIR, self.user_id, 'documents')
        if not os.path.exists(self.user_path):
            os.makedirs(self.user_path)

    def file_path(self, filename):
        return os.path.join(self.user_path, filename)


class DocumentPostgreCRUD(BaseCrud):

    def __init__(self, session: Session, user):
        super().__init__(user)
        self.session = session

    # CREATE
    def new_document(self, filename: str):
        new_obj = Document(
            name=filename,
            filepath=self.file_path(filename),
            user_id=self.user_id,
            upload_date=str(datetime.now()),
            change_date=str(datetime.now()),
            pipeline=[],
            column_marks={}
        )
        self.session.add(new_obj)
        self.session.commit()

    # READ
    def read_document_by_name(self, filename: str):
        filepath = self.file_path(filename)
        document = self.session.query(Document.id, Document.name, Document.upload_date, Document.change_date, Document.pipeline,
                                      Document.column_marks).filter(Document.filepath == filepath).first()
        if document is not None:
            return document
        return None

    def read_all_documents_by_user(self):
        return self.session.query(Document.name, Document.upload_date, Document.change_date).\
            filter(Document.user_id == self.user_id).all()

    # UPDATE
    def update_document(self, filename: str, query: Dict):
        filepath = self.file_path(filename)
        if query.get('name', None):
            query['filepath'] = self.file_path(query['name'])
        self.session.query(Document).filter(Document.filepath == filepath).update(query)
        self.session.commit()

    # DELETE
    def delete_document(self, filename: str):
        filepath = self.file_path(filename)
        self.session.query(Document).filter(Document.filepath == filepath).delete()
        self.session.commit()


class DocumentFileCRUD(BaseCrud):

    # CREATE
    def upload_document(self, filename: str, file):
        csv_path = self.file_path(filename)
        with open(csv_path, 'wb') as buffer:
            shutil.copyfileobj(file, buffer)

    # READ
    def read_document(self, filename: str) -> pd.DataFrame:
        """ DEV USE: Read CSV and return DataFrame object"""
        csv_path = self.file_path(filename)
        data = pd.read_csv(csv_path)
        return data

    def download_document(self, filename: str):
        csv_path = self.file_path(filename)
        return FileResponse(path=csv_path, filename=filename, media_type='text/csv')

    # UPDATE
    def update_document(self, filename: str, data: pd.DataFrame):
        """ DEV USE: Save DataFrame object in CSV format"""
        data.to_csv(os.path.join(self.user_path, filename), index=False)

    def rename_document(self, filename: str, new_filename: str):
        old_path = self.file_path(filename)
        new_path = self.file_path(new_filename)
        shutil.move(old_path, new_path)

    # DELETE
    def delete_document(self, filename: str):
        os.remove(self.file_path(filename))
