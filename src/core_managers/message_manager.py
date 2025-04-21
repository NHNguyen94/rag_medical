from typing import List

from llama_index.core.base.llms.types import ChatMessage, MessageRole


class MessageManager:
    def __init__(self):
        pass

    def construct_message(
        self,
        message: str,
        response: str = "N/A",
        closest_documents: List[str] = List["doc_1"],
        predicted_topic: str = "topic_1",
        recommended_questions: List[str] = List["question_1"],
        predicted_emotion: str = "happy",
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
