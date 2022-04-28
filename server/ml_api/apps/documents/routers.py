from fastapi import APIRouter, Depends, UploadFile, File, status
from fastapi.responses import JSONResponse

from ml_api.common.database.db_deps import get_db
from ml_api.apps.users.routers import current_active_user
from ml_api.apps.users.schemas import UserDB
from ml_api.apps.documents.services import DocumentService
from ml_api.apps.documents.schemas import DocumentDB, ColumnMarks

documents_crud_router = APIRouter(
    prefix="/document",
    tags=["Document Utils"],
    responses={404: {"description": "Not found"}}
)


@documents_crud_router.post("")
def load_document(filename: str, file: UploadFile = File(...), db: get_db = Depends(),
                  user: UserDB = Depends(current_active_user)):
    result = DocumentService(db, user).upload_document_to_db(file=file.file, filename=filename)
    if result:
        return JSONResponse(status_code=status.HTTP_200_OK, content=f"The document '{filename}' successfully added")
    return JSONResponse(status_code=status.HTTP_409_CONFLICT, content=f"The document name '{result}' is already taken")


@documents_crud_router.get("")
def read_document(filename: str, db: get_db = Depends(), user: UserDB = Depends(current_active_user)):
    result = DocumentService(db, user).read_document_from_db(filename)
    if result is None:
        return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content="No such csv document")
    else:
        return JSONResponse(status_code=status.HTTP_200_OK, content=result.head(100).to_json())


@documents_crud_router.get("/info")
def read_document_info(filename: str, db: get_db = Depends(), user: UserDB = Depends(current_active_user)):
    result = DocumentService(db, user).read_document_info(filename=filename)
    return result


@documents_crud_router.put("/pipeline")
def apply_pipeline_to_csv(document_from: str, document_to: str, db: get_db = Depends(), user: UserDB = Depends(current_active_user)):
    pipeline = DocumentService(db, user).read_pipeline(document_from)
    DocumentService(db, user).apply_pipeline_to_csv(filename=document_to, pipeline=pipeline)
    return "OK"


@documents_crud_router.get("/all")
def read_all_user_documents(db: get_db = Depends(), user: UserDB = Depends(current_active_user)):
    result = DocumentService(db, user).read_documents_info()
    return result


@documents_crud_router.get("/download")
def download_document(filename: str, db: get_db = Depends(), user: UserDB = Depends(current_active_user)):
    result = DocumentService(db, user).download_document_from_db(filename)
    return result


@documents_crud_router.put("/rename")
def rename_document(filename: str, new_filename: str, db: get_db = Depends(),
                    user: UserDB = Depends(current_active_user)):
    DocumentService(db, user).rename_document(filename, new_filename)
    return {"filename": new_filename}


@documents_crud_router.delete("")
def delete_document(filename: str, db: get_db = Depends(), user: UserDB = Depends(current_active_user)):
    DocumentService(db, user).delete_document_from_db(filename)
    return {"filename": filename}


@documents_crud_router.get("/columns")
def get_column_names(filename: str, db: get_db = Depends(), user: UserDB = Depends(current_active_user)):
    result = DocumentService(db, user).read_documents_columns(filename)
    return result


@documents_crud_router.put("/column_marks")
def save_column_marks(filename: str, column_marks: ColumnMarks, db: get_db = Depends(),
                      user: UserDB = Depends(current_active_user)):
    result = DocumentService(db, user).update_column_marks(filename, column_marks)
    return result


@documents_crud_router.get("/column_marks")
def read_column_marks(filename: str, db: get_db = Depends(), user: UserDB = Depends(current_active_user)):
    result = DocumentService(db, user).read_column_marks(filename)
    return result


### ---------------------------------------------UNCHECKED--------------------------------------------------------------


documents_method_router = APIRouter(
    prefix="/document/process",
    tags=["Document Methods"],
    responses={404: {"description": "Not found"}}
)


@documents_method_router.put("/duplicates")
def remove_duplicates(filename: str, db: get_db = Depends(), user: UserDB = Depends(current_active_user)):
    DocumentService(db, user).remove_duplicates(filename)
    return {"filename": filename}


@documents_method_router.put("/drop_na")
def remove_duplicates(filename: str, db: get_db = Depends(), user: UserDB = Depends(current_active_user)):
    DocumentService(db, user).drop_na(filename)
    return {"filename": filename}


@documents_method_router.put("/HZR_outliers_OneClassSVM")
def outliers_OneClassSVM(filename: str, iters: float, db: get_db = Depends(),
                         user: UserDB = Depends(current_active_user)):
    DocumentService(db, user).outliers_OneClassSVM(filename, iters)
    return {"filename": filename}


@documents_method_router.put("/HZR_outlier_interquartile_distance")
def outlier_interquartile_distance(filename: str, low_quantile: float, up_quantile: float, coef: float,
                                   db: get_db = Depends(), user: UserDB = Depends(current_active_user)):
    DocumentService(db, user).outlier_interquartile_distance(filename, low_quantile, up_quantile, coef)
    return {"filename": filename}


@documents_method_router.put("/HZR_standartize_features")
def standartize_features(filename: str, db: get_db = Depends(), user: UserDB = Depends(current_active_user)):
    DocumentService(db, user).standardize_features(filename)
    return {"filename": filename}


@documents_method_router.put("/fs_select_k_best")
def fs_select_k_best(filename: str, db: get_db = Depends(), user: UserDB = Depends(current_active_user)):
    DocumentService(db, user).fs_select_k_best(filename)
    return {"filename": filename}


@documents_method_router.put("/outlier_three_sigma")
def outlier_three_sigma(filename: str, db: get_db = Depends(), user: UserDB = Depends(current_active_user)):
    DocumentService(db, user).outlier_three_sigma(filename)
    return {"filename": filename}


@documents_method_router.put("/miss_insert_mean_mode")
def miss_insert_mean_mode(filename: str, db: get_db = Depends(), user: UserDB = Depends(current_active_user)):
    DocumentService(db, user).miss_insert_mean_mode(filename)  # Границу вводит юзер
    return {"filename": filename}
