from abc import ABC
from typing import List
from typing import Optional
from pydantic import PrivateAttr
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.storage.chat_store import BaseChatStore

from src.database.models import ChatHistory
from src.database.service_manager import ServiceManager


class ChatStoreManager(BaseChatStore, ABC):
    _db_service_manager: ServiceManager = PrivateAttr()
    _n_chats: int = PrivateAttr()

    def __init__(self):
        super().__init__()
        self._db_service_manager = ServiceManager()
        self._n_chats = 10

    async def aget_messages(self, key: str) -> List[ChatHistory]:
        has_user_id = await self._db_service_manager.check_existing_user_id(key)
        if not has_user_id:
            await self._db_service_manager.append_user_id(key)
            # Add the first message to the chat history
            await self._db_service_manager.append_chat_history(
                user_id=key,
                message="",
                response="",
                nearest_documents=[],
                predicted_topic="",
                recommended_questions=[],
                predicted_emotion="",
            )
        messages = await self._db_service_manager.get_n_chat_history(key, self._n_chats)
        return messages

    async def async_add_message(
        self,
        key: str,
        message: ChatMessage,
    ) -> None:
        metadata = message.additional_kwargs
        await self._db_service_manager.append_chat_history(
            user_id=key,
            message=message.content,
            response=metadata["response"],
            nearest_documents=metadata["nearest_documents"],
            predicted_topic=metadata["predicted_topic"],
            recommended_questions=metadata["recommended_questions"],
            predicted_emotion=metadata["predicted_emotion"],
        )

    def get_messages(self, key: str) -> None:
        pass

    def add_message(self, key: str, message: ChatMessage) -> None:
        pass

    def set_messages(self, key: str, messages: List[ChatMessage]) -> None:
        pass

    def delete_messages(self, key: str) -> None:
        pass

    def delete_message(self, key: str, idx: int) -> None:
        pass

    def delete_last_message(self, key: str) -> None:
        pass

    def get_keys(self) -> List[str]:
        return []
