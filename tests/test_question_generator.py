import pytest
import numpy as np
import json
from pathlib import Path
import faiss
from unittest.mock import Mock, patch

from src.pipelines.question_recommendation.question_generator import QuestionGenerator


@pytest.fixture
def sample_data():
    # Create sample questions and embeddings
    questions = [
        "What are the symptoms of diabetes?",
        "How is blood pressure measured?",
        "What causes heart disease?"
    ]
    embeddings = np.random.rand(3, 768).astype('float32')  # 3 questions, 768-dim embeddings
    return questions, embeddings


@pytest.fixture
def setup_files(tmp_path, sample_data):
    # Create temporary directory structure
    output_dir = tmp_path / "processed"
    output_dir.mkdir(parents=True)

    questions, embeddings = sample_data

    # Create FAISS index
    index = faiss.IndexFlatL2(768)
    index.add(embeddings)
    faiss.write_index(index, str(output_dir / "faiss_index.bin"))

    # Create questions mapping
    questions_mapping = {i: q for i, q in enumerate(questions)}
    with open(output_dir / "questions_mapping.json", 'w') as f:
        json.dump(questions_mapping, f)

    return output_dir, index, questions_mapping


@pytest.fixture
def question_generator(setup_files):
    output_dir, index, _ = setup_files
    return QuestionGenerator(
        faiss_index=index,
        output_dir=str(output_dir)
    )


class TestQuestionGenerator:
    def test_initialization(self, question_generator, setup_files):
        output_dir, _, _ = setup_files
        assert question_generator.faiss_index is not None
        assert question_generator.output_dir == Path(output_dir)
        assert question_generator.questions_mapping_path == Path(output_dir) / "questions_mapping.json"

    def test_get_similar_questions(self, question_generator, sample_data):
        _, embeddings = sample_data
        query_embedding = embeddings[0].reshape(1, -1)  # Use first embedding as query

        similar_questions = question_generator.get_similar_questions(query_embedding)

        assert isinstance(similar_questions, list)
        assert len(similar_questions) <= 4  # Default k=4
        assert all(isinstance(q, str) for q in similar_questions)

    @patch('src.core_managers.agent_manager.AgentManager.aget_stream_response')
    def test_generate_questions_with_llm(self, mock_aget_stream_response, question_generator):
        mock_aget_stream_response.return_value = "Question 1\nQuestion 2\nQuestion 3"

        context = "Diabetes is a metabolic disease that causes high blood sugar"
        num_questions = 2
        existing_questions = ["What is diabetes?"]

        questions = question_generator.generate_questions_with_llm(
            context=context,
            num_questions=num_questions,
            existing_questions=existing_questions
        )

        assert isinstance(questions, list)
        assert len(questions) == num_questions
        assert all(isinstance(q, str) for q in questions)
        mock_aget_stream_response.assert_called_once()

    @patch('src.question_generator.QuestionGenerator.get_similar_questions')
    @patch('src.question_generator.QuestionGenerator.generate_questions_with_llm')
    def test_generate_follow_up_questions(
            self,
            mock_generate_questions_with_llm,
            mock_get_similar_questions,
            question_generator
    ):
        mock_get_similar_questions.return_value = ["Similar Question 1", "Similar Question 2"]
        mock_generate_questions_with_llm.return_value = ["New Question 1", "New Question 2"]

        context = "Diabetes symptoms include frequent urination and increased thirst"
        questions = question_generator.generate_follow_up_questions(context)

        assert isinstance(questions, list)
        assert len(questions) == 4  # Default num_questions=4
        mock_get_similar_questions.assert_called_once()

        # Test when similar questions are not enough
        mock_get_similar_questions.return_value = ["Similar Question 1"]
        questions = question_generator.generate_follow_up_questions(context)
        mock_generate_questions_with_llm.assert_called_once()