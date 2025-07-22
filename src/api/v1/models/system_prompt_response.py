from typing import Dict

from pydantic import BaseModel


class SystemPromptResponse(BaseModel):
    system_prompt: Dict
