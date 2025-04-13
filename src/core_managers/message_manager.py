from llama_index.core.base.llms.types import ChatMessage, MessageRole


class MessageManager:
    def __init__(self, user_id: str):
        self.chat_history = []
        self.user_id = user_id
        # Handle system message later
        self.system_message = self._construct_message(
            message="You are a helpful assistant",
            role=MessageRole.SYSTEM
        )

    def _append_message(self, message: ChatMessage):
        self.chat_history.append(message)

    def _construct_message(self, message: str, role: str) -> ChatMessage:
        return ChatMessage.from_str(role=role, content=message)

    def construct_user_message(self, message: str) -> ChatMessage:
        user_msg = self._construct_message(
            message=message,
            role=MessageRole.USER
        )
        self._append_message(user_msg)
        return user_msg

    def get_chat_history(self) -> list[ChatMessage]:
        return self.chat_history
