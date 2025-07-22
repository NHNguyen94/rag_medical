from pydantic import BaseModel


class TranscribeResponse(BaseModel):
    transcription: str
