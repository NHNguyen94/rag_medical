from typing import Optional, List

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
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.chat_history_manager = ChatHistoryManager()
        self.vector_store_manager = VectorStoreManager(user_id)
        # Handle the vector store initialization later
        self.prompt_manager = PromptManager(chat_bot_config.DEFAULT_PROMPT_PATH)
        self.system_prompt_template = self.prompt_manager.make_system_prompt(
            self.prompt_manager.get_system_prompt()
        )
        self.index = self.vector_store_manager.build_or_load_index("indices")
        self.agent = AgentManager(
            # TODO: Load index at the lifespan of the app
            index=self.index,
            chat_model=chat_bot_config.DEFAULT_CHAT_MODEL,
            system_prompt_template=self.system_prompt_template,
            reasoning_effort=self.prompt_manager.get_reasoning_effort(),
            temperature=self.prompt_manager.get_temperature(),
        )
        self.response_manager = ResponseManager()

    async def achat(
        self,
        message: str,
        closest_documents: Optional[List] = [],
        predicted_topic: Optional[str] = "",
        recommended_questions: Optional[List] = [],
        predicted_emotion: Optional[str] = "",
    ) -> str:
        chat_history = await self.chat_history_manager.get_chat_history(self.user_id)
        # print(f"Chat history: {chat_history}")
        response = await self.agent.aget_stream_response(message, chat_history)
        response_str = await self.response_manager.parse_stream_response(response)
        # TODO: Implement the features here
        await self.chat_history_manager.append_chat_history_to_db(
            user_id=self.user_id,
            message=message,
            response=response_str,
            closest_documents=closest_documents,
            predicted_topic=predicted_topic,
            recommended_questions=recommended_questions,
            predicted_emotion=predicted_emotion,
        )
        return response_str
