from src.core_managers.audio_manager import AudioManager

import pytest

class TestAudioManager():
    audio_path = "tests/resources/test_audio.wav"

    def test_transcribe(self):
        audio_manager = AudioManager()
        transcription = audio_manager.transcribe(self.audio_path)
        print(transcription)
        assert isinstance(transcription, str)

    @pytest.mark.asyncio
    async def test_atranscribe(self):
        audio_manager = AudioManager()
        transcription = await audio_manager.atranscribe(self.audio_path)
        print(transcription)
        assert isinstance(transcription, str)