from dataclasses import dataclass
from typing import Literal

from llama_index.core import PromptTemplate
from llama_index.core.prompts import PromptType

from src.utils.helpers import load_yml_configs


@dataclass
class PromptConfig:
    system_prompt: str
    reasoning_effort: Literal["low", "medium", "high"]
    temperature: float


class PromptManager:
    def __init__(self, prompt_path: str):
        self.prompt = PromptConfig(**load_yml_configs(prompt_path))

    def get_system_prompt(self) -> str:
        return self.prompt.system_prompt

    def get_reasoning_effort(self) -> Literal["low", "medium", "high"]:
        return self.prompt.reasoning_effort

    def get_temperature(self) -> float:
        return self.prompt.temperature

    def make_system_prompt(self, system_prompt: str) -> PromptTemplate:
        return PromptTemplate(template=system_prompt, prompt_type=PromptType.CUSTOM)
