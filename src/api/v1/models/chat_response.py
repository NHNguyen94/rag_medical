from typing import List

from pydantic import BaseModel


class ChatResponse(BaseModel):
    response: str
    nearest_documents: List[str]
