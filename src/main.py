import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.v1.api import router as v1_router
from src.core_managers.vector_store_manager import VectorStoreManager
from src.database.models import create_tables
from src.services.emotion_recognition_service import EmotionRecognitionService
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
    app.state.emotion_recognition_service = emotion_recognition_service

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

    model, vocab = await emotion_recognition_service.async_load_model()
    app.state.emotion_model = model
    app.state.emotion_vocab = vocab

    await create_tables()
    yield


app = FastAPI(lifespan=lifespan)

app.include_router(v1_router)
