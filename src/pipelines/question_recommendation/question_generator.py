from typing import List, Dict
import random
import numpy as np
from pathlib import Path

from src.core_managers.agent_manager import AgentManager
from src.core_managers.vector_store_manager import VectorStoreManager
from src.utils.enums import QuestionRecommendConfig


class QuestionGenerator:

    def __init__(
            self,
            faiss_index,
            questions_mapping: Dict[str, str],
            output_dir: str = QuestionRecommendConfig.PROCESSED_DATA_DIR,
            model_name: str = "gpt-3.5-turbo",
            temperature: float = 0.7
    ):
        self.output_dir = Path(output_dir)
        self.faiss_index = faiss_index
        self.questions_mapping = questions_mapping
        self.model_name = model_name
        self.temperature = temperature

    def get_similar_questions(self, query_embedding, k: int = 5) -> List[str]:
        """Retrieve similar questions using FAISS."""
        if isinstance(query_embedding, list):
            query_embedding = np.array(query_embedding)

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        distances, indices = self.faiss_index.search(query_embedding.astype('float32'), k)
        # Get questions from mapping
        updated_indices = indices[0][1:]
        similar_questions = [self.questions_mapping[idx] for idx in updated_indices
                           if idx in self.questions_mapping]

        # Remove duplicates while preserving order
        results = []
        for q in similar_questions:
            if q not in results:
                results.append(q)

        return results[:k]

    def generate_follow_up_questions(
            self,
            question_embedding: np.ndarray,
            num_questions: int = 4
    ) -> List[str]:
        """Generate follow-up questions using FAISS."""
        similar_questions = self.get_similar_questions(question_embedding, k=num_questions)
        return similar_questions[:num_questions]