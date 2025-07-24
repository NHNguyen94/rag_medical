from typing import Optional

from fastapi import APIRouter, Request
from loguru import logger
import json

from src.api.v1.models import (
    ChatRequest,
    BaseChatRequest,
    ChatResponse,
    TextToSpeechRequest,
    TextToSpeechResponse,
    TranscribeRequest,
    TranscribeResponse,
    FeedbackRequest,
    FeedbackResponse,
)
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


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(feedback_request: FeedbackRequest):
    try:
        from src.database.models import ResponseFeedback
        from src.database.session_manager import SessionManager
        
        session_manager = SessionManager()
        async with session_manager.get_async_session() as session:
            feedback = ResponseFeedback(
                user_id=feedback_request.user_id,
                message=feedback_request.message,
                response=feedback_request.response,
                feedback_type=feedback_request.feedback_type,
            )
            session.add(feedback)
            await session.commit()
            
        logger.info(f"Feedback submitted: {feedback_request.feedback_type} by user {feedback_request.user_id}")
        return FeedbackResponse(
            success=True,
            message=f"Feedback submitted successfully"
        )
    except Exception as e:
        logger.error(f"Error submitting feedback: {e}")
        return FeedbackResponse(
            success=False,
            message=f"Error submitting feedback: {str(e)}"
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
    disable_emotion_recognition: Optional[bool] = False,
):
    cache_service = CacheService()

    cache_request = {
        "chat_request": chat_request.model_dump(),
        "force_use_tools": force_use_tools,
        "use_cot": use_cot,
        "disable_emotion_recognition": disable_emotion_recognition,
    }
    
    # Add cache_buster to cache_request if present in chat_request
    if hasattr(chat_request, 'cache_buster') and chat_request.cache_buster:
        cache_request["cache_buster"] = chat_request.cache_buster
        logger.info(f"[DEBUG] Cache buster: {chat_request.cache_buster}")

    logger.info(f"[DEBUG] Cache key: {json.dumps(cache_request, sort_keys=True)}")
    logger.info(f"[DEBUG] User message: {chat_request.message}")

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
        model_name=chat_request.model_name,
        disable_emotion_recognition=disable_emotion_recognition,
        language=chat_request.language,
    )

    cache_service.cache_request_and_response(
        request=cache_request,
        response=response.model_dump(),
    )
    logger.info("Returning new response")
    return response


async def get_response(
    chat_request: BaseChatRequest,
    request: Request,
    force_use_tools: bool,
    use_cot: bool,
    customized_sys_prompt_path: Optional[str] = None,
    customize_index_path: Optional[str] = None,
    model_name: Optional[str] = None,
    disable_emotion_recognition: Optional[bool] = False,
    language: Optional[str] = "English",
):
    try:
        predicted_emotion = ""
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
            customized_sys_prompt_path=customized_sys_prompt_path,
            customize_index_path=customize_index_path,
            model_name=model_name,
            language=language,
            user_emotion=predicted_emotion if not disable_emotion_recognition and predicted_emotion else None,
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

        if disable_emotion_recognition:
            predicted_emotion = ""
            logger.info(f"Emotion recognition DISABLED for user {chat_request.user_id}")
        else:
            predicted_emotion = emotion_recognition_service.predict(
                text=chat_msg,
                model=emotion_model,
                vocab=emotion_vocab,
            )
            logger.info(f"Emotion recognition ENABLED for user {chat_request.user_id}, predicted emotion: {predicted_emotion}")

        logger.info(f"[DEBUG] Passing emotion to ChatBotService: {predicted_emotion}")

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
        logger.error(f"[ERROR] Exception in get_response: {e}")
        fallback_response = ChatResponse(
            response="Sorry, something went wrong. Please try again later.",
            nearest_documents=[],
        )
        return fallback_response
