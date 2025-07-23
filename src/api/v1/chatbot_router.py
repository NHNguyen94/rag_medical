from typing import Optional

from fastapi import APIRouter, Request
from loguru import logger

from src.api.v1.models import (
    ChatRequest,
    BaseChatRequest,
    ChatResponse,
    TextToSpeechRequest,
    TextToSpeechResponse,
    TranscribeRequest,
    TranscribeResponse
)
from src.api.v1.models.ai_question_request import AiquestionRequest
from src.api.v1.models.ai_question_response import AiquestionResponse
from src.services.audio_service import AudioService
from src.services.cache_service import CacheService
from src.services.chat_bot_service import ChatBotService
from src.utils.enums import ChatBotConfig

router = APIRouter(tags=["chatbot"])


@router.post("/text_to_speech", response_model=TextToSpeechResponse)
async def text_to_speech(
    text_to_speech_request: TextToSpeechRequest,
):
    audio_service = AudioService()
    audio_file = await audio_service.atext_to_speech(
        text=text_to_speech_request.text, output_path=text_to_speech_request.audio_path
    )
    logger.info(f"Generated audio file: {audio_file}")
    return TextToSpeechResponse(
        audio_path=text_to_speech_request.audio_path,
    )


@router.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(
    transcribe_request: TranscribeRequest,
):
    audio_service = AudioService()
    transcribed_msg = await audio_service.atranscribe(transcribe_request.audio_file)
    logger.info(f"Transcribed message: {transcribed_msg}")
    return TranscribeResponse(
        transcription=transcribed_msg,
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(
    chat_request: ChatRequest,
    request: Request,
    # Disable tool to manually retrieve documents
    force_use_tools: bool = False,
    use_cot: bool = True,
    customized_sys_prompt_path: Optional[str] = None,
    customize_index_path: Optional[str] = None,
):
    cache_service = CacheService()

    cache_request = {
        "chat_request": chat_request.model_dump(),
        "force_use_tools": force_use_tools,
        "use_cot": use_cot,
    }

    cached_response = cache_service.get_cached_response(cache_request)
    if cached_response:
        logger.info("Returning cached response")
        return ChatResponse(**cached_response)

    response = await get_response(
        chat_request=chat_request,
        request=request,
        force_use_tools=force_use_tools,
        use_cot=use_cot,
        customized_sys_prompt_path=customized_sys_prompt_path,
        customize_index_path=customize_index_path,
    )

@router.post("/ai-question", response_model=AiquestionResponse)
async def get_ai_question(
        question_request: AiquestionRequest,
        request: Request
):
    question_recommendation_service = request.app.state.question_recomm_service
    question = question_recommendation_service.ai_predict(question_request.topic)

    return AiquestionResponse(recommended_question=question)


async def get_response(
    chat_request: BaseChatRequest,
    request: Request,
    force_use_tools: bool,
    use_cot: bool,
    customized_sys_prompt_path: Optional[str] = None,
    customize_index_path: Optional[str] = None,
):
    try:
        # TODO: Implement the all the features here
        emotion_recognition_service = request.app.state.emotion_recognition_service
        emotion_model = request.app.state.emotion_model
        emotion_vocab = request.app.state.emotion_vocab

        topic_clustering_service = request.app.state.topic_clustering_service
        topic_cluster_model = request.app.state.topic_cluster_model

        use_qr = chat_request.use_qr
        question_recommendation_service = request.app.state.question_recomm_service

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
            customized_sys_prompt_path=customized_sys_prompt_path,
            customize_index_path=customize_index_path,
        )

        chat_msg = chat_request.message

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
        nearest_nodes = await chat_bot_service.retrieve_related_nodes(message=chat_msg)
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

        qr_model_name = f"qr_{predicted_topic_no}"
        print(f"qr_model_name: {qr_model_name}")
        qr_model = getattr(request.app.state, qr_model_name)
        if use_qr:
            question_recommendations = question_recommendation_service.predict(
                input_question=chat_msg,
                model=qr_model
            )
        else:
            question_recommendations = []

        await chat_bot_service.append_history(
            message=chat_msg,
            response_str=response,
            nearest_documents=nearest_documents,
            predicted_emotion=str(predicted_emotion.item()),
            recommended_questions=question_recommendations
        )

        response = ChatResponse(
            response=response,
            nearest_documents=nearest_documents,
            recommended_questions=question_recommendations
        )

        return response
    except Exception as e:
        print(f"Error when getting the response: {e}")
