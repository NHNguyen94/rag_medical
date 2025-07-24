from typing import List

from llama_index.core.base.llms.types import MessageRole, ChatMessage

from src.database.models import ChatHistory
from src.database.service_manager import ServiceManager


class ChatHistoryManager:
    def __init__(self):
        self.db_service_manager = ServiceManager()
        self.n_chats = 10

    async def _check_user_id(self, user_id: str) -> bool:
        return await self.db_service_manager.check_existing_user_id(user_id)

    async def _append_user_id(self, user_id: str) -> None:
        await self.db_service_manager.append_user(user_id)

    async def _append_first_message_to_db(self, user_id: str) -> None:
        await self.db_service_manager.append_chat_history(
            user_id=user_id,
            message="",
            response="",
            nearest_documents=[],
            predicted_topic="",
            recommended_questions=[],
            predicted_emotion="",
        )

    async def _get_chat_history_from_db(self, user_id: str) -> List[ChatHistory]:
        return await self.db_service_manager.get_n_chat_history(user_id, self.n_chats)

    def _construct_chat_history(
        self, chat_history: List[ChatHistory]
    ) -> List[ChatMessage]:
        messages = []
        for chat in chat_history:
            messages.append(
                ChatMessage(
                    role=MessageRole.USER,
                    content=chat.message,
                    additional_kwargs={
                        "response": chat.response,
                        "nearest_documents": chat.nearest_documents,
                        "predicted_topic": chat.predicted_topic,
                        "recommended_questions": chat.recommended_questions,
                        "predicted_emotion": chat.predicted_emotion,
                    },
                )
            )
            messages.append(
                ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content=chat.response,
                    additional_kwargs={
                        "response": "",
                        "nearest_documents": [],
                        "predicted_topic": "",
                        "recommended_questions": [],
                        "predicted_emotion": "",
                    },
                )
            )
        return messages

    async def get_chat_history(self, user_id: str) -> List[ChatMessage]:
        has_user_id = await self._check_user_id(user_id)
        if not has_user_id:
            await self._append_user_id(user_id)
        chat_history = await self._get_chat_history_from_db(user_id)
        return self._construct_chat_history(chat_history)

    async def append_chat_history_to_db(
        self,
        user_id: str,
        message: str,
        response: str,
        nearest_documents: List[str],
        predicted_topic: str,
        recommended_questions: List[str],
        predicted_emotion: str,
    ) -> None:
        await self.db_service_manager.append_chat_history(
            user_id=user_id,
            message=message,
            response=response,
            nearest_documents=nearest_documents,
            predicted_topic=predicted_topic,
            recommended_questions=recommended_questions,
            predicted_emotion=predicted_emotion,
        )

    async def delete_chat_history(self, user_id: str) -> None:
        await self.db_service_manager.delete_chat_history(user_id)

    async def get_user_chat_history(self, user_id: str, limit: int = 10) -> None:
        history = await self.db_service_manager.get_latest_chat_history(user_id, limit)
        return history

    async def delete_single_chat_message(self, user_id: str) -> None:
        await self.db_service_manager.delete_single_chat(user_id)
