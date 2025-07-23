from pydantic import BaseModel


class HistoryDeleteResponse(BaseModel):
    user_id: str
    message: str = "Chat history deleted successfully."
