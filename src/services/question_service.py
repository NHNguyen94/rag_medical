from src.ml_models import flan_t5


class QuestionService:
    def __init__(self):
        self.model = flan_t5.FlanT5()

    def get_follow_up_question(self, input_question: str, domain: str) -> list[str]:
        return self.model.recommend(input_question, domain)
