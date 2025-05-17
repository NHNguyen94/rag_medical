from typing import Any

from fastapi import APIRouter, Request

from src.api.v1.models.chat_request import ChatRequest
from src.api.v1.models.chat_response import ChatResponse
from src.services.chat_bot_service import ChatBotService
from src.utils.enums import ChatBotConfig

router = APIRouter(tags=["chatbot"])


@router.post("/chat", response_model=ChatResponse)
async def chat(
    chat_request: ChatRequest, request: Request, force_use_tools: bool = True
):
    try:
        # TODO: Implement the all the features here
        lstm_model = request.app.state.emotion_recognition_service
        index_cancer = request.app.state.index_cancer
        index_diabetes = request.app.state.index_diabetes
        index_genetic = request.app.state.index_genetic
        index_hormone = request.app.state.index_hormone
        index_heart_lung_blood = request.app.state.index_heart_lung_blood
        index_neuro_disorders_and_stroke = (
            request.app.state.index_neuro_disorders_and_stroke
        )
        index_senior_health = request.app.state.index_senior_health
        index_others = request.app.state.index_others

        domain = ChatBotConfig.DOMAIN_MAPPING[chat_request.selected_domain]
        print(f"Domain: {domain}")
        match domain:
            case ChatBotConfig.CANCER:
                index = index_cancer
            case ChatBotConfig.DIABETES:
                index = index_diabetes
            case ChatBotConfig.DISEASE_CONTROL_AND_PREVENTION:
                index = index_diabetes
            case ChatBotConfig.GENETIC_AND_RARE_DISEASES:
                index = index_genetic
            case ChatBotConfig.GROWTH_HORMONE_RECEPTOR:
                index = index_hormone
            case ChatBotConfig.HEART_LUNG_AND_BLOOD:
                index = index_heart_lung_blood
            case ChatBotConfig.NEUROLOGICAL_DISORDERS_AND_STROKE:
                index = index_neuro_disorders_and_stroke
            case ChatBotConfig.SENIOR_HEALTH:
                index = index_senior_health
            case ChatBotConfig.OTHERS:
                index = index_others
            case _:
                raise ValueError(f"Invalid domain: {domain}")

        chat_bot_service = ChatBotService(
            user_id=chat_request.user_id, index=index, force_use_tools=force_use_tools
        )

        predicted_emotion = lstm_model.predict([chat_request.message])
        nearest_documents = await chat_bot_service.aget_nearest_documents(
            message=chat_request.message
        )

        response = await chat_bot_service.achat(
            message=chat_request.message,
            customer_emotion=int(predicted_emotion),
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
