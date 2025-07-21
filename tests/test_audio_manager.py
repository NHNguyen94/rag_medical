from src.core_managers.audio_manager import AudioManager

import pytest


class TestAudioManager:
    audio_path = "tests/resources/test_audio.wav"
    language = "en"

    def test_transcribe(self):
        audio_manager = AudioManager(self.language)
        transcription = audio_manager.transcribe(self.audio_path)
        print(transcription)
        assert isinstance(transcription, str)

    @pytest.mark.asyncio
    async def test_atranscribe(self):
        audio_manager = AudioManager(self.language)
        transcription = await audio_manager.atranscribe(self.audio_path)
        print(transcription)
        assert isinstance(transcription, str)

    def test_text_to_speech(self):
        audio_manager = AudioManager(self.language)
        output_path = "tests/resources/test_output.wav"
        text = "Hello, how are you today? How's the weather today?"
        audio_manager.text_to_speech(text, output_path)

    @pytest.mark.asyncio
    async def test_atext_to_speech(self):
        audio_manager = AudioManager(self.language)
        output_path = "tests/resources/test_output_async.wav"
        text = "Hello, how are you today? How's the weather today?"
        await audio_manager.atext_to_speech(text, output_path)
