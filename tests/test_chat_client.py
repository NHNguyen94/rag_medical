from typing import Dict

from src.clients.chat_client import ChatClient
import pytest


class TestChatClient:
    client = ChatClient(base_url="http://localhost:8000")

    def test_chat(self):
        user_id = "test_user"
        message = "Hello, how are you?"
        selected_domain = "Others"
        response = self.client.chat(
            user_id=user_id, message=message, selected_domain=selected_domain
        )
        assert isinstance(response, Dict)
        assert len(response) > 0

    def test_voice_chat(self):
        user_id = "test_user"
        audio_file = "tests/resources/test_audio.wav"
        selected_domain = "Others"
        response = self.client.voice_chat(
            user_id=user_id, audio_file=audio_file, selected_domain=selected_domain
        )
        assert isinstance(response, Dict)
        assert len(response) > 0

    # @pytest.mark.asyncio
    # async def test_achat(self):
    #     user_id = "test_user"
    #     message = "Hello, how are you?"
    #     selected_domain = "All"
    #     response = await self.client.achat(
    #         user_id=user_id, message=message, selected_domain=selected_domain
    #     )
    #     print(f"Response async: {response}")
    #     assert isinstance(response, str)
    #     assert len(response) > 0
