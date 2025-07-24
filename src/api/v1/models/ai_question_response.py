from pydantic import BaseModel

class AiquestionResponse(BaseModel):
    recommended_question: str