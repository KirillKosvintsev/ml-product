from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from bunnet import init_bunnet
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError

from ml_api import config
from ml_api.apps.users.routers import users_router
from ml_api.apps.dataframes.routers import (
    dataframes_file_router,
    dataframes_metadata_router,
    dataframes_content_router,
    dataframes_methods_router,
    dataframes_specs_router)
from ml_api.apps.ml_models.routers import (
    models_file_router,
    models_metadata_router,
    models_processing_router,
    models_specs_router)
from ml_api.apps.training_reports.routers import (
    reports_router, reports_specs_router)
from ml_api.apps.jobs.routers import (
    jobs_router, jobs_specs_router)

from ml_api.apps.users.model import User
from ml_api.apps.dataframes.model import DataFrameMetadata
from ml_api.apps.ml_models.model import ModelMetadata
from ml_api.apps.training_reports.model import Report
from ml_api.apps.jobs.model import BackgroundJob

app = FastAPI(
    title=config.PROJECT_NAME,
    version=config.VERSION,
    docs_url=config.DOCS_URL,
    openapi_url=config.OPENAPI_URL,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# @app.exception_handler(ApplyFunctionException)
# async def unicorn_exception_handler(request: Request, exc: ApplyFunctionException):
#     return JSONResponse(
#         status_code=status.HTTP_404_NOT_FOUND,
#         content=str(exc),
#     )


@app.on_event("startup")
def app_init():
    """Initialize application services"""
    app.db = MongoClient(config.MONGO_DATABASE_URI)[config.MONGO_DB_NAME]
    try:
        init_bunnet(app.db,
            document_models=[User, DataFrameMetadata, ModelMetadata, Report,
                             BackgroundJob])
    except ServerSelectionTimeoutError as sste:
        print(sste)


api_router = APIRouter(prefix=config.API_PREFIX)
api_router.include_router(users_router)
api_router.include_router(dataframes_file_router)
api_router.include_router(dataframes_metadata_router)
api_router.include_router(dataframes_content_router)
api_router.include_router(dataframes_methods_router)
api_router.include_router(models_file_router)
api_router.include_router(models_metadata_router)
api_router.include_router(models_processing_router)
api_router.include_router(reports_router)
api_router.include_router(jobs_router)
api_router.include_router(dataframes_specs_router)
api_router.include_router(models_specs_router)
api_router.include_router(reports_specs_router)
api_router.include_router(jobs_specs_router)
app.include_router(api_router)
