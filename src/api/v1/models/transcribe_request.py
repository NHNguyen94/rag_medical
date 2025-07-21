from pydantic import BaseModel

class TranscribeRequest(BaseModel):
    audio_file: str