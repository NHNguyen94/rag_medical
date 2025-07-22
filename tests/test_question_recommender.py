import pytest
import asyncio

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

    @pytest.mark.asyncio
    async def test_async_cancer_question_recommender(self):
        model = await qr_service.async_load_model(0)  # or pass mapped path
        question = "What are the causes cancer?"
        output = qr_service.predict(question, model)
        print(f"response1: {output}")
        assert isinstance(output, list)

    @pytest.mark.asyncio
    async def test_async_diabetes_question_recommender(self):
        model = await qr_service.async_load_model(1)  # or mapped path
        question = "What are the causes diabetes?"
        output = qr_service.predict(question, model)
        print(f"response2: {output}")
        assert isinstance(output, list)
