from typing import List

from sqlmodel import select

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

    async def check_existing_user_id(self, username: str) -> bool:
        async with self.session_manager.get_async_session() as session:
            result = await session.execute(
                select(Users).where(Users.username == username)
            )
            return result.scalars().first() is not None

    async def get_hashed_password(self, username: str) -> str:
        async with self.session_manager.get_async_session() as session:
            result = await session.execute(
                select(Users.hashed_password).where(Users.username == username)
            )
            return result.scalars().first()

    async def append_user(self, username: str, hashed_password: str = None) -> Users:
        async with self.session_manager.get_async_session() as session:
            if hashed_password is None:
                random_id = str(get_unique_id())
                hashed_password = hash_string(random_id)
            user = Users(username=username, hashed_password=hashed_password)
            session.add(user)
            await session.commit()
            await session.refresh(user)
            return user
