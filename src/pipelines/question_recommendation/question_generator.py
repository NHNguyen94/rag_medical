from typing import List, Dict
import openai
import numpy as np
from pathlib import Path

from src.core_managers.agent_manager import AgentManager
from src.core_managers.vector_store_manager import VectorStoreManager


class QuestionGenerator:

    def __init__(
            self,
            faiss_index,
            questions_mapping: Dict[str, str],
            output_dir: str = "../../data/processed",
            model_name: str = "gpt-3.5-turbo",
            temperature: float = 0.7
    ):
        self.output_dir = Path(output_dir)
        self.faiss_index = faiss_index
        self.questions_mapping = questions_mapping
        self.model_name = model_name
        self.temperature = temperature

        # self.questions_mapping_path = self.output_dir / "questions_mapping.json"
        # self.agent_manager = AgentManager(
        #     index=faiss_index,
        #     chat_model=model_name,
        #     temperature=temperature
        # )

    def get_similar_questions(self, query_embedding, k: int = 4) -> List[str]:
        """Retrieve similar questions using FAISS."""
        """Retrieve similar questions using FAISS."""
        # Ensure query_embedding is the right shape
        if isinstance(query_embedding, list):
            query_embedding = np.array(query_embedding)

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Search in FAISS index
        distances, indices = self.faiss_index.search(query_embedding.astype('float32'), k)

        # Get questions using the mapping
        return [self.questions_mapping[str(idx)] for idx in indices[0] if str(idx) in self.questions_mapping]

    def generate_questions_with_llm(
            self,
            context: str,
            num_questions: int,
            existing_questions: List[str]
    ) -> List[str]:
        """Generate additional questions using LLM."""
        prompt = f"""
        Given the following medical context:
        {context}

        And these existing questions:
        {existing_questions}

        Generate {num_questions} new, diverse follow-up questions that:
        1. Are medically relevant
        2. Are different from the existing questions
        3. Cover different aspects of the topic
        4. Are clear and well-formed

        Return only the questions, one per line.
        """

        response = self.agent_manager.aget_stream_response(prompt)
        questions = [q.strip() for q in response.split('\n') if q.strip()]
        return questions[:num_questions]

    def generate_follow_up_questions(
            self,
            question_embedding: np.ndarray,
            num_questions: int = 4
    ) -> List[str]:
        """Generate follow-up questions using both FAISS and LLM."""
        # Get similar questions from FAISS
        # similar_questions = self.get_similar_questions(context, k=num_questions)
        similar_questions = self.get_similar_questions(question_embedding, k=num_questions)

        # If we don't have enough questions, generate more with LLM
        # if len(similar_questions) < num_questions:
        #     additional_questions = self.generate_questions_with_llm(
        #         context,
        #         num_questions - len(similar_questions),
        #         similar_questions
        #     )
        #     similar_questions.extend(additional_questions)

        return similar_questions[:num_questions]