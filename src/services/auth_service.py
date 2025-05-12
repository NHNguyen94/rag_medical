from passlib.context import CryptContext

from src.database.service_manager import ServiceManager


class AuthenticationService:
    def __init__(self):
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.db_service_manager = ServiceManager()

    async def login(self, username: str, password: str) -> bool:
        existing_user = await self.db_service_manager.check_existing_user_id(username)
        if not existing_user:
            raise ValueError("User does not exist")

        hashed_password = await self.db_service_manager.get_hashed_password(username)
        if not self.pwd_context.verify(password, hashed_password):
            raise ValueError("Invalid password")

        return True

    async def register(self, username: str, password: str) -> None:
        existing_user = await self.db_service_manager.check_existing_user_id(username)
        if existing_user:
            raise ValueError("User already exists")

        hashed_password = self.pwd_context.hash(password)
        await self.db_service_manager.append_user(username, hashed_password)
