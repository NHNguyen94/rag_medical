from fastapi import APIRouter

router = APIRouter(tags=["chatbot"])


@router.post("/chat")
def chat():
    pass


@router.post("/summarize")
def summarize():
    pass
