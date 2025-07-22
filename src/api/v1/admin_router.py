from fastapi import APIRouter

from src.api.v1.models import SystemPromptRequest, SystemPromptResponse
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
