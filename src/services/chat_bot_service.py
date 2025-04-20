from src.core_managers import (
    AgentManager,
    VectorStoreManager,
    ResponseManager,
    PromptManager,
)
from src.core_managers.chat_history_manager import ChatHistoryManager
from src.core_managers.chat_store_manager import ChatStoreManager
from src.core_managers.memory_manager import MemoryManager
from src.core_managers.message_manager import MessageManager
from src.utils.enums import ChatBotConfig

chat_bot_config = ChatBotConfig()


class ChatBotService:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.memory = MemoryManager().initialize_chat_memory(
            self.user_id, ChatStoreManager()
        )
        self.message_manager = MessageManager()
        self.chat_history_manager = ChatHistoryManager()
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
            # memory=self.memory,
        )
        self.response_manager = ResponseManager()

    async def achat(
            self,
            message: str,
    ) -> str:
        chat_history = await self.chat_history_manager.get_chat_history(self.user_id)
        print(f"Chat history: {chat_history}")
        response = await self.agent.aget_stream_response(message, chat_history)
        response_str = await self.response_manager.parse_stream_response(response)
        return response_str
