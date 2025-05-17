from typing import Any

from fastapi import APIRouter
from src.api.v1.models.user_login_request import UserLoginRequest
from src.services.auth_service import AuthenticationService

router = APIRouter(tags=["auth"])


@router.post("/login")
async def login(
    request: UserLoginRequest,
) -> Any:
    auth_service = AuthenticationService()
    try:
        await auth_service.login(request.username, request.password)
        return {"message": "Login successful"}
    except ValueError as e:
        return {"error": str(e)}


@router.post("/register")
async def register(
    request: UserLoginRequest,
) -> Any:
    auth_service = AuthenticationService()
    try:
        await auth_service.register(request.username, request.password)
        return {"message": "Registration successful"}
    except ValueError as e:
        return {"error": str(e)}
