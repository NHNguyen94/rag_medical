from fastapi import APIRouter

from src.api.v1.chatbot_router import router as chatbot_router

router = APIRouter(prefix="/v1")

router.include_router(chatbot_router, prefix="/chatbot")
