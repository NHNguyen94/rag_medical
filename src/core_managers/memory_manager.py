from typing import Optional

from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.storage.chat_store import BaseChatStore


class MemoryManager:
    def __init__(self, token_limit: Optional[int] = 3000):
        self.token_limit = token_limit

    def initialize_chat_memory(
        self, user_id: str, chat_store: BaseChatStore
    ) -> ChatMemoryBuffer:
        chat_memory = ChatMemoryBuffer.from_defaults(
            token_limit=self.token_limit,
            chat_store=chat_store,
            chat_store_key=user_id,
        )
        return chat_memory
