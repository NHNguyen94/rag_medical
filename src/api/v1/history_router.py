from fastapi import APIRouter
from typing import List

from src.api.v1.models import (
    HistoryDeleteResponse,
    HistoryDeleteRequest,
    ChatHistoryResponse,
)
from src.core_managers.chat_history_manager import ChatHistoryManager

router = APIRouter(tags=["history"])


@router.post("/delete_chat_history", response_model=HistoryDeleteResponse)
async def delete_chat_history(history_delete_request: HistoryDeleteRequest):
    chat_history_manager = ChatHistoryManager()
    await chat_history_manager.delete_chat_history(history_delete_request.user_id)
    return HistoryDeleteResponse(
        user_id=history_delete_request.user_id,
        message=f"Chat history for user {history_delete_request.user_id} deleted successfully.",
    )


@router.get("/chat-history/{user_id}", response_model=List[ChatHistoryResponse])
async def get_chat_history(user_id: str, limit: int = 10):
    chat_history_manager = ChatHistoryManager()
    history = await chat_history_manager.get_user_chat_history(user_id, limit)
    return history


@router.delete("/chat-history/message/{chat_id}")
async def delete_chat_message(chat_id: str):
    chat_history_manager = ChatHistoryManager()
    await chat_history_manager.delete_single_chat_message(chat_id)
    return {"status": "success"}
