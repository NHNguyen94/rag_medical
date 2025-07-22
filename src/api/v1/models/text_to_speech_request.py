from pydantic import BaseModel


class TextToSpeechRequest(BaseModel):
    text: str
    audio_path: str
