import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.v1.api import router as v1_router
from src.core_managers.vector_store_manager import VectorStoreManager
from src.database.models import create_tables
from src.services.emotion_recognition_service import EmotionRecognitionService
from src.services.question_service import QuestionService
from src.services.topic_clustering_service import TopicClusteringService
from src.utils.enums import IngestionConfig, ChatBotConfig

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

vt_store = VectorStoreManager()

# Disable parallelism for tokenizers to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# https://fastapi.tiangolo.com/advanced/events/#startup-and-shutdown-together
async def load_model_background(
    app: FastAPI, emotion_recognition_service: EmotionRecognitionService
) -> None:
    model, vocab = await emotion_recognition_service.async_load_model()
    app.state.emotion_model = model
    app.state.emotion_vocab = vocab


@asynccontextmanager
async def lifespan(app: FastAPI):
    emotion_recognition_service = EmotionRecognitionService()
    topic_clustering_service = TopicClusteringService()
    question_recomm_service = QuestionService()
    app.state.emotion_recognition_service = emotion_recognition_service
    app.state.topic_clustering_service = topic_clustering_service
    app.state.question_recomm_service = question_recomm_service

    index_path = IngestionConfig.INDEX_PATH
    index_cancer = vt_store.build_or_load_index(f"{index_path}/{ChatBotConfig.CANCER}")
    index_diabetes = vt_store.build_or_load_index(
        f"{index_path}/{ChatBotConfig.DIABETES}"
    )
    index_disease_control_and_prevention = vt_store.build_or_load_index(
        f"{index_path}/{ChatBotConfig.DISEASE_CONTROL_AND_PREVENTION}"
    )
    index_genetic = vt_store.build_or_load_index(
        f"{index_path}/{ChatBotConfig.GENETIC_AND_RARE_DISEASES}"
    )
    index_hormone = vt_store.build_or_load_index(
        f"{index_path}/{ChatBotConfig.GROWTH_HORMONE_RECEPTOR}"
    )
    index_heart_lung_blood = vt_store.build_or_load_index(
        f"{index_path}/{ChatBotConfig.HEART_LUNG_AND_BLOOD}"
    )
    index_neuro_disorders_and_stroke = vt_store.build_or_load_index(
        f"{index_path}/{ChatBotConfig.NEUROLOGICAL_DISORDERS_AND_STROKE}"
    )
    index_senior_health = vt_store.build_or_load_index(
        f"{index_path}/{ChatBotConfig.SENIOR_HEALTH}"
    )
    index_others = vt_store.build_or_load_index(f"{index_path}/{ChatBotConfig.OTHERS}")
    app.state.index_cancer = index_cancer
    app.state.index_diabetes = index_diabetes
    app.state.index_disease_control_and_prevention = (
        index_disease_control_and_prevention
    )
    app.state.index_genetic = index_genetic
    app.state.index_hormone = index_hormone
    app.state.index_heart_lung_blood = index_heart_lung_blood
    app.state.index_neuro_disorders_and_stroke = index_neuro_disorders_and_stroke
    app.state.index_senior_health = index_senior_health
    app.state.index_others = index_others

    emotion_model, emotion_vocab = await emotion_recognition_service.async_load_model()
    app.state.emotion_model = emotion_model
    app.state.emotion_vocab = emotion_vocab


    topic_cluster_model = await topic_clustering_service.async_load_model()
    app.state.topic_cluster_model = topic_cluster_model

    qr_cancer = await question_recomm_service.async_load_model(0)
    qr_diabetes = await question_recomm_service.async_load_model(1)
    # qr_disease_cntrl_prev = await question_recomm_service.async_load_model(2)
    # qr_genetic_hormone = await question_recomm_service.async_load_model(3)
    # qr_growth_hormone = await question_recomm_service.async_load_model(4)
    # qr_heart_lung = await question_recomm_service.async_load_model(5)
    # qr_neurological = await question_recomm_service.async_load_model(6)
    qr_senior_health = await question_recomm_service.async_load_model(7)
    # qr_other = await question_recomm_service.async_load_model(8)

    app.state.qr_0 = qr_cancer
    app.state.qr_1 = qr_diabetes
    # app.state.qr_2 = qr_disease_cntrl_prev
    # app.state.qr_3 = qr_genetic_hormone
    # app.state.qr_4 = qr_growth_hormone
    # app.state.qr_5 = qr_heart_lung
    # app.state.qr_6 = qr_neurological
    app.state.qr_7 = qr_senior_health
    # app.state.qr_8 = qr_other

    await create_tables()
    yield


app = FastAPI(lifespan=lifespan)

app.include_router(v1_router)
