from fastapi import APIRouter, Request

from src.api.v1.models.chat_request import ChatRequest
from src.api.v1.models.chat_response import ChatResponse
from src.services.chat_bot_service import ChatBotService

router = APIRouter(tags=["chatbot"])


@router.post("/chat", response_model=ChatResponse)
async def chat(chat_request: ChatRequest, request: Request):
    lstm_model = request.app.state.emotion_recognition_service
    predicted_emotion = lstm_model.predict([chat_request.message])

    chat_bot_service = ChatBotService(user_id=chat_request.user_id)
    response = await chat_bot_service.achat(
        message=chat_request.message, predicted_emotion=str(predicted_emotion.item())
    )
    response = ChatResponse(response=response)
    # print(f"Chat response: {response}")

    return response


@router.post("/summarize")
def summarize():
    pass
