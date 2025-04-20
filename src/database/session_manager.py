import os

import dotenv
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

dotenv.load_dotenv()


class SessionManager:
    def __init__(self):
        self.engine = create_async_engine(
            os.getenv("DATABASE_URL"),
            future=True,
            # echo=True,
        )
        self._async_sessionmaker = sessionmaker(
            bind=self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
            autocommit=False,
        )

    def close(self):
        self.engine.dispose()

    def connect(self):
        return self.engine.connect()

    async def get_async_session(self):
        async with self._async_sessionmaker() as session:
            yield session
