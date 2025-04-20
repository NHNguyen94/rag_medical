from datetime import datetime
from typing import List, Optional

from sqlmodel import SQLModel, Field, Column, JSON

from src.database.session_manager import SessionManager

# https://github.com/fastapi/sqlmodel/issues/178
class ChatHistory(SQLModel, table=True):
    __tablename__ = "chat_history"

    id: Optional[str] = Field(primary_key=True, default=None)
    user_id: str = Field(nullable=False)
    message: str = Field(nullable=False)
    response: str = Field(nullable=False)
    closest_documents: List[str] = Field(sa_column=Column(JSON))
    predicted_topic: str = Field(nullable=False)
    recommended_questions: List[str] = Field(sa_column=Column(JSON))
    predicted_emotion: str = Field(nullable=False)
    utc_timestamp: datetime = Field(nullable=False)


def get_engine():
    session_manager = SessionManager()
    return session_manager.engine


def create_tables():
    engine = get_engine()
    SQLModel.metadata.create_all(engine)
