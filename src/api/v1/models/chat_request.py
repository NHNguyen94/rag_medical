from pydantic import BaseModel


class BaseChatRequest(BaseModel):
    user_id: str
    selected_domain: str
    use_qr: bool = True
    customized_sys_prompt_path: str | None = None
    customize_index_path: str | None = None


class ChatRequest(BaseChatRequest):
    message: str


class VoiceChatRequest(BaseChatRequest):
    audio_file: str
