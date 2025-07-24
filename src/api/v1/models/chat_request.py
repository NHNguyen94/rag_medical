from pydantic import BaseModel


class BaseChatRequest(BaseModel):
    user_id: str
    selected_domain: str
    use_qr: bool = True

class ChatRequest(BaseChatRequest):
    message: str


class VoiceChatRequest(BaseChatRequest):
    audio_file: str
