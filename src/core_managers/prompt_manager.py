from dataclasses import dataclass
from typing import Literal, Optional, List

from llama_index.core import PromptTemplate
from llama_index.core.prompts import PromptType

from src.utils.enums import ChatBotConfig
from src.utils.helpers import load_yml_configs

chat_bot_config = ChatBotConfig


@dataclass
class PromptConfig:
    system_prompt: str
    reasoning_effort: Literal["low", "medium", "high"]
    temperature: float
    similarity_top_k: int


class PromptManager:
    def __init__(self, prompt_path: str):
        self.prompt = PromptConfig(**load_yml_configs(prompt_path))

    def get_system_prompt(self) -> str:
        return self.prompt.system_prompt

    def get_reasoning_effort(self) -> Literal["low", "medium", "high"]:
        return self.prompt.reasoning_effort

    def get_temperature(self) -> float:
        return self.prompt.temperature

    def get_similarity_top_k(self) -> int:
        return self.prompt.similarity_top_k

    def make_system_prompt(self, system_prompt: str) -> PromptTemplate:
        return PromptTemplate(template=system_prompt, prompt_type=PromptType.CUSTOM)

    def add_prompts_for_additional_services(
        self,
        current_prompt: str,
        customer_emotion: Optional[int] = None,
        synthesized_response: Optional[str] = None,
        nearest_documents: Optional[List[str]] = None,
    ) -> str:
        # Put all services that require to update the system prompt here
        if customer_emotion:
            emotion_str = chat_bot_config.EMOTION_MAPPING[customer_emotion]
            emotion_prompt = f"""
            -----------
            This is the emotion of the user,
            use this to adjust your tone of voice,

            User emotion:
            {emotion_str}
            -----------
            """
            current_prompt = current_prompt + emotion_prompt

        if nearest_documents:
            documents_prompt = f"""
            -----------
            These are the retrieved documents,
            summarize the content of these documents,
            and use the information to answer the user question,
            
            Retrieved documents:
            {nearest_documents}
            -----------
            """
            current_prompt = current_prompt + documents_prompt

        if synthesized_response:
            reasoning_prompt = f"""
            -----------
            This is the synthesized response from the retrieved documents,
            use this to refine your final response,

            Synthesized response:
            {synthesized_response}
            -----------
            """
            current_prompt = current_prompt + reasoning_prompt

        return current_prompt
