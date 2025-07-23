from pydantic import BaseModel


class HistoryDeleteRequest(BaseModel):
    user_id: str
