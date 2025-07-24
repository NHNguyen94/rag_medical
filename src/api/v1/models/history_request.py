from pydantic import BaseModel
from datetime import datetime

class HistoryDeleteRequest(BaseModel):
    user_id: str


class ChatHistoryResponse(BaseModel):
    id: str
    user_id: str
    message: str
    response: str
    created_at: datetime