from datetime import datetime
from typing import List
from uuid import UUID, uuid4

from sqlmodel import SQLModel, Field, Column, JSON, Index

from src.database.session_manager import SessionManager
from src.utils.date_time_manager import DateTimeManager


# https://github.com/fastapi/sqlmodel/issues/178
class ChatHistory(SQLModel, table=True):
    __tablename__ = "chat_history"
    __table_args__ = (
        Index("ix_chat_history_user_id", "user_id"),
        Index("ix_chat_history_created_at", "created_at"),
    )

    id: UUID = Field(primary_key=True, default_factory=uuid4)
    user_id: str = Field(nullable=False)
    message: str = Field(nullable=False)
    response: str = Field(nullable=False)
    nearest_documents: List[str] = Field(sa_column=Column(JSON))
    predicted_topic: str = Field(nullable=False)
    recommended_questions: List[str] = Field(sa_column=Column(JSON))
    predicted_emotion: str = Field(nullable=False)
    created_at: datetime = Field(
        nullable=False, default=DateTimeManager.get_current_utc_time()
    )


class Users(SQLModel, table=True):
    __tablename__ = "users"

    user_id: str = Field(primary_key=True, default_factory=None)
    created_at: datetime = Field(
        nullable=False, default=DateTimeManager.get_current_utc_time()
    )


def get_engine():
    session_manager = SessionManager()
    return session_manager.engine


async def create_tables():
    async with get_engine().begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
