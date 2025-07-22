from src.services.question_service import QuestionService
from src.utils.enums import QuestionRecommendConfig

qr_config = QuestionRecommendConfig()

def run_inference(text: str) -> None:
    qr_service = QuestionService()
    model_path = qr_config.MODEL_PATH
    model = qr_service.load_model(1)
    followup_questions = qr_service.predict(text, model)
    for ques in followup_questions:
        print(ques)


if __name__ == "__main__":
    texts = [
        "What are the symptoms of diabetes?"
    ]

    for question in texts:
        run_inference(question)
