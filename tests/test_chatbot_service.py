from src.services.chat_bot_service import ChatBotService
import pytest


@pytest.mark.asyncio
class TestChatBotService:
    chat_bot_service = ChatBotService("test_user")

    @pytest.mark.skip(reason="Costs tokens to run, skip by default")
    async def test_achat(self):
        response = await self.chat_bot_service.achat("Hello")
        assert isinstance(response, str)
