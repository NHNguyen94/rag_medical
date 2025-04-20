from dataclasses import dataclass

from typing import Literal

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
