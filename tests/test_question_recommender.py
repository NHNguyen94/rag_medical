from src.ml_models.flan_t5 import FlanT5
from src.services.question_service import QuestionService


qr_service = QuestionService()
class TestQuestionRecommender:

    def test_cancer_question_recommender(self):
        model = qr_service.load_model(0)
        question = "What are the causes cancer?"
        output = qr_service.predict(question, model)
        print(f"response1: {output} ")
        assert isinstance(output, list)

    def test_diabetes_question_recommender(self):
        model = qr_service.load_model(1)
        question = "What are the causes diabetes?"
        output = qr_service.predict(question, model)
        print(f"response1: {output} ")
        assert isinstance(output, list)
