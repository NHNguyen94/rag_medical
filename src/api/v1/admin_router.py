from fastapi import APIRouter
from loguru import logger

from src.api.v1.models import (
    SystemPromptRequest,
    SystemPromptResponse,
    IngestCustomFileRequest,
    IngestCustomFileResponse,
)
from src.services.ingestion_service import IngestionService
from src.utils.enums import IngestionConfig
from src.utils.helpers import write_dict_to_yaml, load_yml_configs

router = APIRouter(tags=["admin"])


@router.post("/update_system_prompt", response_model=SystemPromptResponse)
def update_system_prompt(system_prompt_request: SystemPromptRequest):
    write_dict_to_yaml(
        system_prompt_request.model_dump(), system_prompt_request.yml_file
    )

    return SystemPromptResponse(
        system_prompt=load_yml_configs(system_prompt_request.yml_file)
    )


@router.post("/ingest_custom_file", response_model=IngestCustomFileResponse)
def ingest_custom_file(ingest_custom_file_request: IngestCustomFileRequest):
    ingestion_service = IngestionService()
    ingestion_service.ingest_data(
        data_path=ingest_custom_file_request.file_dir_path,
        index_path=ingest_custom_file_request.index_dir_path,
        col_name_to_ingest=IngestionConfig.COL_NAME_TO_INGEST,
    )

    return IngestCustomFileResponse(
        index_dir_path=ingest_custom_file_request.index_dir_path,
    )
