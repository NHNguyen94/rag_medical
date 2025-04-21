from typing import List

from llama_index.core.base.llms.types import ChatMessage, MessageRole


class MessageManager:
    def __init__(self):
        pass

    def construct_message(
        self,
        message: str,
        response: str,
        closest_documents: List[str],
        predicted_topic: str,
        recommended_questions: List[str],
        predicted_emotion: str,
        role: str = MessageRole.USER,
    ) -> ChatMessage:
        additional_kwargs = {
            "response": response,
            "closest_documents": closest_documents,
            "predicted_topic": predicted_topic,
            "recommended_questions": recommended_questions,
            "predicted_emotion": predicted_emotion,
        }
        return ChatMessage.from_str(
            role=role, content=message, additional_kwargs=additional_kwargs
        )
