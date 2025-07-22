from typing import Optional, List

from llama_index.core.base.response.schema import RESPONSE_TYPE
from llama_index.core.indices.base import BaseIndex

from src.core_managers import (
    AgentManager,
    VectorStoreManager,
    ResponseManager,
    PromptManager,
)
from src.core_managers.chat_history_manager import ChatHistoryManager
from src.utils.enums import ChatBotConfig

chat_bot_config = ChatBotConfig()


class ChatBotService:
    def __init__(
            self,
            user_id: str,
            index: BaseIndex,
            force_use_tools: bool,
            use_cot: bool,
    ):
        self.user_id = user_id
        self.chat_history_manager = ChatHistoryManager()
        self.vector_store_manager = VectorStoreManager()
        if use_cot:
            self.prompt_manager = PromptManager(chat_bot_config.COT_PROMPT_PATH)
        else:
            self.prompt_manager = PromptManager(chat_bot_config.DEFAULT_PROMPT_PATH)
        self.system_prompt_template = self.prompt_manager.make_system_prompt(
            self.prompt_manager.get_system_prompt()
        )
        self.index = index
        self.agent = AgentManager(
            index=self.index,
            chat_model=chat_bot_config.DEFAULT_CHAT_MODEL,
            system_prompt_template=self.system_prompt_template,
            reasoning_effort=self.prompt_manager.get_reasoning_effort(),
            temperature=self.prompt_manager.get_temperature(),
            similarity_top_k=self.prompt_manager.get_similarity_top_k(),
            force_use_tools=force_use_tools,
        )
        self.response_manager = ResponseManager()
        self.use_cot = use_cot

    def update_system_prompt(
            self,
            customer_emotion: Optional[int] = None,
            synthesized_response: Optional[str] = None,
            nearest_documents: Optional[List[str]] = None,
    ) -> None:
        current_prompt = self.prompt_manager.get_system_prompt()
        new_prompt = self.prompt_manager.add_prompts_for_additional_services(
            current_prompt=current_prompt,
            customer_emotion=customer_emotion,
            synthesized_response=synthesized_response,
            nearest_documents=nearest_documents,
        )
        self.system_prompt_template = self.prompt_manager.make_system_prompt(new_prompt)

    async def achat(
            self,
            message: str,
            customer_emotion: Optional[int] = None,
            synthesized_response: Optional[str] = None,
            nearest_documents: Optional[List[str]] = None,
    ) -> str:
        self.update_system_prompt(
            customer_emotion=customer_emotion,
            synthesized_response=synthesized_response,
            nearest_documents=nearest_documents,
        )
        # print(f"\nNew system prompt: {self.system_prompt_template}\n")
        chat_history = await self.chat_history_manager.get_chat_history(self.user_id)
        # print(f"Chat history: {chat_history}")
        response = await self.agent.aget_stream_response(message, chat_history)
        response_str = await self.response_manager.parse_stream_response(response)
        return response_str

    async def retrieve_related_nodes(self, message: str) -> RESPONSE_TYPE:
        return await self.agent.retrieve_related_nodes(message)

    async def aget_nearest_documents(self, nearest_nodes: RESPONSE_TYPE) -> List[str]:
        return await self.agent.aget_nearest_documents(nearest_nodes)

    async def asynthesize_response(
            self,
            message: str,
            nearest_nodes: RESPONSE_TYPE,
    ) -> str:
        response = await self.agent.asynthesize_response(message, nearest_nodes)
        return str(response)

    async def append_history(
            self,
            message: str,
            response_str: str,
            nearest_documents: Optional[List] = [],
            predicted_topic: Optional[str] = "",
            recommended_questions: Optional[List] = [],
            predicted_emotion: Optional[str] = "",
    ) -> None:
        await self.chat_history_manager.append_chat_history_to_db(
            user_id=self.user_id,
            message=message,
            response=response_str,
            nearest_documents=nearest_documents,
            predicted_topic=predicted_topic,
            recommended_questions=recommended_questions,
            predicted_emotion=predicted_emotion,
        )
