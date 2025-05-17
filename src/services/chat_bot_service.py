from typing import Optional, List

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
    def __init__(self, user_id: str, index: BaseIndex, force_use_tools: bool = True):
        self.user_id = user_id
        self.chat_history_manager = ChatHistoryManager()
        self.vector_store_manager = VectorStoreManager()
        # Handle the vector store initialization later
        self.prompt_manager = PromptManager(chat_bot_config.DEFAULT_PROMPT_PATH)
        self.system_prompt_template = self.prompt_manager.make_system_prompt(
            self.prompt_manager.get_system_prompt()
        )
        # self.index = self.vector_store_manager.build_or_load_index("src/indices")
        self.index = index
        self.agent = AgentManager(
            index=self.index,
            chat_model=chat_bot_config.DEFAULT_CHAT_MODEL,
            system_prompt_template=self.system_prompt_template,
            reasoning_effort=self.prompt_manager.get_reasoning_effort(),
            temperature=self.prompt_manager.get_temperature(),
            force_use_tools=force_use_tools,
        )
        self.response_manager = ResponseManager()

    def update_system_prompt(self, predicted_customer_emotion: int) -> None:
        emotion_str = chat_bot_config.EMOTION_MAPPING[predicted_customer_emotion]
        current_prompt = self.prompt_manager.get_system_prompt()
        current_prompt = current_prompt + f"""
        The customer is feeling: {emotion_str},
        please adjust your response accordingly
        """
        self.system_prompt_template = self.prompt_manager.make_system_prompt(
            current_prompt
        )

    async def achat(
            self,
            message: str,
            customer_emotion: Optional[int] = None,
    ) -> str:
        # Put all services that require to update the system prompt here
        if customer_emotion:
            self.update_system_prompt(customer_emotion)
            # print(f"\nNew system prompt: {self.system_prompt_template}\n")

        chat_history = await self.chat_history_manager.get_chat_history(self.user_id)
        # print(f"Chat history: {chat_history}")
        response = await self.agent.aget_stream_response(message, chat_history)
        response_str = await self.response_manager.parse_stream_response(response)
        return response_str

    async def aget_nearest_documents(self, message: str) -> List[str]:
        matched_documents = await self.agent.aget_nearest_documents(message)
        return matched_documents

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
