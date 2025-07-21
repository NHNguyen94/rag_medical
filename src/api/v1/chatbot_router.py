from typing import Any

from fastapi import APIRouter, Request
from loguru import logger

from src.api.v1.models.chat_request import ChatRequest, VoiceChatRequest, BaseChatRequest
from src.api.v1.models.chat_response import ChatResponse
from src.services.chat_bot_service import ChatBotService
from src.services.question_service import QuestionService
from src.utils.enums import ChatBotConfig

router = APIRouter(tags=["chatbot"])


@router.post("/chat", response_model=ChatResponse)
async def chat(
    chat_request: ChatRequest,
    request: Request,
    # Disable tool to manually retrieve documents
    force_use_tools: bool = False,
    use_cot: bool = True,
):
    return await get_response(
        chat_request=chat_request,
        request=request,
        force_use_tools=force_use_tools,
        use_cot=use_cot,
    )

@router.post("/voice_chat", response_model=ChatResponse)
async def voice_chat(
    voice_chat_request: VoiceChatRequest,
    request: Request,
    # Disable tool to manually retrieve documents
    force_use_tools: bool = False,
    use_cot: bool = True,
):
    return await get_response(
        chat_request=voice_chat_request,
        request=request,
        force_use_tools=force_use_tools,
        use_cot=use_cot,
    )

async def get_response(
    chat_request: BaseChatRequest,
    request: Request,
    force_use_tools: bool,
    use_cot: bool,
):
    try:
        # TODO: Implement the all the features here
        emotion_recognition_service = request.app.state.emotion_recognition_service
        emotion_model = request.app.state.emotion_model
        emotion_vocab = request.app.state.emotion_vocab

        topic_clustering_service = request.app.state.topic_clustering_service
        topic_cluster_model = request.app.state.topic_cluster_model

        index_cancer = request.app.state.index_cancer
        index_diabetes = request.app.state.index_diabetes
        index_disease_control_and_prevention = (
            request.app.state.index_disease_control_and_prevention
        )
        index_genetic = request.app.state.index_genetic
        index_hormone = request.app.state.index_hormone
        index_heart_lung_blood = request.app.state.index_heart_lung_blood
        index_neuro_disorders_and_stroke = (
            request.app.state.index_neuro_disorders_and_stroke
        )
        index_senior_health = request.app.state.index_senior_health
        index_others = request.app.state.index_others

        domain = ChatBotConfig.DOMAIN_MAPPING[chat_request.selected_domain]
        match domain:
            case ChatBotConfig.CANCER:
                index = index_cancer
            case ChatBotConfig.DIABETES:
                index = index_diabetes
            case ChatBotConfig.DISEASE_CONTROL_AND_PREVENTION:
                index = index_disease_control_and_prevention
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

        # question_recomend_service = QuestionService()
        chat_bot_service = ChatBotService(
            user_id=chat_request.user_id,
            index=index,
            force_use_tools=force_use_tools,
            use_cot=use_cot,
        )

        if isinstance(chat_request, VoiceChatRequest):
            chat_msg = await chat_bot_service.atranscribe(chat_request.audio_file)
        elif isinstance(chat_request, ChatRequest):
            chat_msg = chat_request.message
        else:
            raise ValueError("Invalid chat request type")

        print(f"\n\n\nChat message: {chat_msg}\n\n\n")

        # Use later for question recommendation
        topic_domain = ChatBotConfig.DOMAIN_ENCODE_MAPPING
        predicted_topic_no = topic_clustering_service.predict(
            chat_msg,
            topic_cluster_model,
        )
        logger.info(f"Predicted topic no: {predicted_topic_no}")
        for k, v in topic_domain.items():
            if v == predicted_topic_no:
                predicted_topic = k
                logger.info(f"Predicted topic: {predicted_topic}")
                break

        predicted_emotion = emotion_recognition_service.predict(
            text=chat_msg,
            model=emotion_model,
            vocab=emotion_vocab,
        )
        nearest_nodes = await chat_bot_service.retrieve_related_nodes(
            message=chat_msg
        )
        nearest_documents = await chat_bot_service.aget_nearest_documents(
            nearest_nodes=nearest_nodes
        )
        synthesized_response = await chat_bot_service.asynthesize_response(
            message=chat_msg,
            nearest_nodes=nearest_nodes,
        )
        response = await chat_bot_service.achat(
            message=chat_msg,
            customer_emotion=int(predicted_emotion),
            nearest_documents=nearest_documents,
            synthesized_response=synthesized_response,
        )

        # follow_up_questions = question_recomend_service.get_follow_up_question(chat_request.message, domain)
        await chat_bot_service.append_history(
            message=chat_msg,
            response_str=response,
            nearest_documents=nearest_documents,
            predicted_emotion=str(predicted_emotion.item()),
            # recommended_questions=follow_up_questions
        )

        response = ChatResponse(
            response=response,
            nearest_documents=nearest_documents,
            # recommended_questions=follow_up_questions
        )

        return response
    except Exception as e:
        print(f"Error when getting the response: {e}")
