from src.core_managers.audio_manager import AudioManager
from src.utils.enums import AudioConfig

audio_config = AudioConfig()


class AudioService:
    def __init__(
        self,
        language: str = audio_config.DEFAULT_LANGUAGE,
        whisper_model_name: str = audio_config.DEFAULT_WHISPER_MODEL,
    ):
        self.audio_manager = AudioManager(
            language=language, whisper_model_name=whisper_model_name
        )

    async def atranscribe(self, audio_path: str) -> str:
        return await self.audio_manager.atranscribe(audio_path)
