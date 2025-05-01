import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, BackgroundTasks

from src.api.v1.api import router as v1_router
from src.database.models import create_tables
from src.services.emotion_recognition_service import EmotionRecognitionService

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Disable parallelism for tokenizers to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# https://fastapi.tiangolo.com/advanced/events/#startup-and-shutdown-together
async def load_model_background(emotion_recognition_service: EmotionRecognitionService):
    await emotion_recognition_service.async_load_model()


@asynccontextmanager
async def lifespan(app: FastAPI):
    emotion_recognition_service = EmotionRecognitionService(use_embedding=True)
    app.state.emotion_recognition_service = emotion_recognition_service

    # Start background task to load the model after the app starts
    background_tasks = BackgroundTasks()
    background_tasks.add_task(load_model_background, emotion_recognition_service)

    await create_tables()
    yield


app = FastAPI(lifespan=lifespan)

app.include_router(v1_router)
