from src.ml_models.flan_t5 import FlanT5
from src.services.question_service import QuestionService


class TestQuestionRecommender:
    model = FlanT5()
    service = QuestionService()

    def test_question_recommender(self):
        question = "How is cancer diagnosed?"
        output = self.model.recommend(question, 'Cancer')
        print(f"response1: {output} ")
        assert isinstance(output, list)

    def test_question_service(self):
        question = "How is cancer diagnosed?"
        follow_up = self.service.get_follow_up_question(question, domain='Cancer')
        print(f"response2: {follow_up} ")
        assert isinstance(follow_up, list)