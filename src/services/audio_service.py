from src.core_managers.audio_manager import AudioManager
from src.utils.enums import GeneralConfig

general_config = GeneralConfig()


class AudioService:
    def __init__(
            self,
            language: str = general_config.DEFAULT_LANGUAGE
    ):
        self.audio_manager = AudioManager(
            language=language
        )

    async def atranscribe(self, audio_path: str) -> str:
        return await self.audio_manager.atranscribe(audio_path)
