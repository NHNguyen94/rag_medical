from pydantic import BaseModel
from typing import Optional


class BaseChatRequest(BaseModel):
    user_id: str
    selected_domain: str

    model_name: Optional[str] = None
    disable_emotion_recognition: Optional[bool] = False
    use_qr: bool = True


class ChatRequest(BaseChatRequest):
    message: str
    cache_buster: Optional[str] = None
    language: Optional[str] = "English"


class VoiceChatRequest(BaseChatRequest):
    audio_file: str

class FeedbackRequest(BaseModel):
    user_id: str
    message: str
    response: str
    feedback_type: str  # "like" or "dislike"

class FeedbackResponse(BaseModel):
    success: bool
    message: str
