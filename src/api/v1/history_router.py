from fastapi import APIRouter

from src.api.v1.models import HistoryDeleteResponse, HistoryDeleteRequest
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
