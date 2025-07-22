from pydantic import BaseModel


class TextToSpeechResponse(BaseModel):
    audio_path: str
