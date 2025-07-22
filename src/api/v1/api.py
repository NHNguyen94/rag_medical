from fastapi import APIRouter

from src.api.v1.admin_router import router as admin_router
from src.api.v1.auth_router import router as auth_router
from src.api.v1.chatbot_router import router as chatbot_router

router = APIRouter(prefix="/v1")

router.include_router(chatbot_router, prefix="/chatbot")
router.include_router(auth_router, prefix="/auth")
router.include_router(admin_router, prefix="/admin")
