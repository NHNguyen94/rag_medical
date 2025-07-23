from typing import List

from sqlmodel import select, desc, delete

from src.database.models import ChatHistory, Users
from src.database.session_manager import SessionManager
from src.utils.helpers import hash_string, get_unique_id


class ServiceManager:
    def __init__(self):
        self.session_manager = SessionManager()

    async def append_chat_history(
        self,
        user_id: str,
        message: str,
        response: str,
        nearest_documents: List[str],
        predicted_topic: str,
        recommended_questions: List[str],
        predicted_emotion: str,
    ) -> ChatHistory:
        async with self.session_manager.get_async_session() as session:
            chat_history = ChatHistory(
                user_id=user_id,
                message=message,
                response=response,
                nearest_documents=nearest_documents,
                predicted_topic=predicted_topic,
                recommended_questions=recommended_questions,
                predicted_emotion=predicted_emotion,
            )
            session.add(chat_history)
            await session.commit()
            await session.refresh(chat_history)
            return chat_history

    # https://docs.sqlalchemy.org/en/20/tutorial/data_select.html
    async def get_n_chat_history(self, user_id: str, n: int) -> List[ChatHistory]:
        async with self.session_manager.get_async_session() as session:
            result = await session.execute(
                select(ChatHistory)
                .where(ChatHistory.user_id == user_id)
                .order_by(ChatHistory.__table__.c.created_at.desc())
                .limit(n)
            )
            return result.scalars().all()

    async def check_existing_user_id(self, hashed_username: str) -> bool:
        async with self.session_manager.get_async_session() as session:
            result = await session.execute(
                select(Users).where(Users.hashed_username == hashed_username)
            )
            return result.scalars().first() is not None

    async def get_hashed_password(self, hashed_username: str) -> str:
        async with self.session_manager.get_async_session() as session:
            result = await session.execute(
                select(Users.hashed_password).where(
                    Users.hashed_username == hashed_username
                )
            )
            return result.scalars().first()

    async def append_user(
        self, hashed_username: str, hashed_password: str = None
    ) -> Users:
        async with self.session_manager.get_async_session() as session:
            if hashed_password is None:
                random_id = str(get_unique_id())
                hashed_password = hash_string(random_id)
            user = Users(
                hashed_username=hashed_username, hashed_password=hashed_password
            )
            session.add(user)
            await session.commit()
            await session.refresh(user)
            return user

    async def delete_chat_history(self, user_id: str) -> None:
        async with self.session_manager.get_async_session() as session:
            await session.execute(
                ChatHistory.__table__.delete().where(ChatHistory.user_id == user_id)
            )
            await session.commit()

    async def get_latest_chat_history(self, user_id: str, limit: int = 10) -> list[dict]:
        async with self.session_manager.get_async_session() as session:
            result = await session.execute(
                select(ChatHistory)
                .where(ChatHistory.user_id == user_id)
                .order_by(desc(ChatHistory.created_at))
                .limit(limit)
            )
            rows = result.scalars().all()
            return [
                {
                    "id": str(row.id),
                    "user_id": str(row.user_id),
                    "message": row.message,
                    "response": row.response,
                    "created_at": row.created_at.isoformat(),
                }
                for row in rows
            ]

    async def delete_single_chat(self, chat_id: str) -> None:
        async with self.session_manager.get_async_session() as session:
            await session.execute(
                delete(ChatHistory).where(ChatHistory.id == chat_id)
            )
            await session.commit()
