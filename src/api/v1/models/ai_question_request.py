from pydantic import BaseModel


class AiquestionRequest(BaseModel):
    user_id: str
    topic: str
