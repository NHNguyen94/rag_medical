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

    def test_transcribe(self):
        audio_file = "tests/resources/test_audio.wav"
        response = self.client.transcribe(audio_file=audio_file)
        print(f"Transcription response: {response}")
        assert isinstance(response, Dict)
        assert "transcription" in response


    def test_text_to_speech(self):
        text = "This is a test for text to speech."
        audio_path = "tests/resources/test_audio_output_chat_client.wav"
        response = self.client.text_to_speech(text=text, audio_path=audio_path)
        print(f"Text to speech response: {response}")
        # assert isinstance(response, Dict)
        # assert "audio_path" in response
        # assert response["audio_path"] == audio_path
