from src.ml_models.flan_t5 import FlanT5
from src.services.question_service import QuestionService\


class TestQuestionRecommender:
    model = FlanT5()
    service = QuestionService()

    def test_question_recommender(self):
        question = "What are the symptoms of covid-19?"
        output = self.model.recommend(question)
        print(f"response1: {output} ")
        assert isinstance(output, str) and len(output) > 5

    def test_question_service(self):
        question = "What are the symptoms of covid-19?"
        follow_up = self.service.get_follow_up_question(question)
        print(f"response2: {follow_up} ")
        assert isinstance(follow_up, str) and len(follow_up) > 5