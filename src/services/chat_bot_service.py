from src.core_managers import (
    AgentManager,
    VectorStoreManager,
    ResponseManager,
    PromptManager,
)
from src.utils.enums import ChatBotConfig

chat_bot_config = ChatBotConfig()


class ChatBotService:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.vector_store_manager = VectorStoreManager(user_id)
        # Handle the vector store initialization later
        self.prompt_manager = PromptManager(chat_bot_config.DEFAULT_PROMPT_PATH)
        self.index = self.vector_store_manager.build_index([])
        self.agent = AgentManager(
            index=self.index,
            chat_model=chat_bot_config.DEFAULT_CHAT_MODEL,
            system_prompt=self.prompt_manager.get_system_prompt(),
            reasoning_effort=self.prompt_manager.get_reasoning_effort(),
            temperature=self.prompt_manager.get_temperature(),
        )
        self.response_manager = ResponseManager()

    async def achat(self, message: str) -> str:
        response = await self.agent.aget_stream_response(message)
        response_str = await self.response_manager.parse_stream_response(response)
        return response_str
