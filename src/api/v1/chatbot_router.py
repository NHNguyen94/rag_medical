from fastapi import APIRouter, Request

from src.api.v1.models.chat_request import ChatRequest
from src.api.v1.models.chat_response import ChatResponse

from src.services.chat_bot_service import ChatBotService

router = APIRouter(tags=["chatbot"])


@router.post("/chat", response_model=ChatResponse)
async def chat(chat_request: ChatRequest, request: Request, force_use_tools: bool = True):
    try:
        # TODO: Implement the all the features here
        lstm_model = request.app.state.emotion_recognition_service
        index = request.app.state.index
        predicted_emotion = lstm_model.predict([chat_request.message])

        chat_bot_service = ChatBotService(
            user_id=chat_request.user_id,
            index=index,
            force_use_tools=force_use_tools
        )
        response = await chat_bot_service.achat(
            message=chat_request.message
        )
        nearest_documents = await chat_bot_service.aget_nearest_documents(
            message=chat_request.message
        )
        await chat_bot_service.append_history(
            message=chat_request.message,
            response_str=response,
            nearest_documents=nearest_documents,
            predicted_emotion=str(predicted_emotion.item()),
        )

        response = ChatResponse(response=response)
        # print(f"Chat response: {response}")

        return response
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        return ChatResponse(response="An error occurred while processing your request.")


@router.post("/summarize")
def summarize():
    pass
