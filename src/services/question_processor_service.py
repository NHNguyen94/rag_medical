from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from pathlib import Path
import json
import torch
import faiss
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from src.core_managers.encoding_manager import EncodingManager
from src.core_managers.vector_store_manager import VectorStoreManager
from src.utils.helpers import clean_text
from src.utils.directory_manager import DirectoryManager
from src.utils.enums import QuestionRecommendConfig


class QuestionDataProcessor:
    def __init__(
        self,
        data_dir: str = None,
        output_dir: str = None,
        embedding_dim: int = 768,
    ):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.embedding_dim = embedding_dim

        # Initialize BERT model for embeddings
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.model = AutoModel.from_pretrained("bert-base-uncased")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def load_datasets(self) -> pd.DataFrame:
        file_path = self.data_dir
        df = pd.read_csv(Path(file_path))
        df["source"] = Path(file_path).stem
        return df

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess the data."""
        df["cleaned_question"] = df["Question"]
        df = df.drop_duplicates(subset=["cleaned_question"])
        df = df.dropna(subset=["cleaned_question"])
        df.to_csv(self.output_dir / "cleaned_dataset.csv", index=False)

        return df

    def create_embeddings(self, questions: List[str]) -> List[float]:
        """Create embeddings for questions using BERT."""
        embeddings = []
        for question in tqdm(questions, desc="Creating embeddings"):
            # Tokenize and create embedding
            inputs = self.tokenizer(
                question, return_tensors="pt", padding=True, truncation=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use [CLS] token embedding
                embedding = (
                    outputs.last_hidden_state[:, 0, :]
                    .cpu()
                    .numpy()[0]
                    .astype("float32")
                )

                if embedding.size != self.embedding_dim:
                    raise ValueError(f"Expected {self.embedding_dim}, got {emb.size}")

                embeddings.append(embedding)

        return np.stack(embeddings, axis=0)

    def build_faiss_index(self, questions: List[str], embeddings: List[List[float]]):
        """Build FAISS index for question retrieval."""
        embeddings_array = np.array(embeddings).astype("float32")

        faiss_index = faiss.IndexFlatL2(self.embedding_dim)
        faiss_index.add(embeddings_array)

        return faiss_index

    def process_datasets(self) -> Dict:
        """Process all datasets and prepare for training."""
        print("Loading datasets...")
        combined_df = self.load_datasets()

        print("Preprocessing data...")
        processed_df = self.preprocess_data(combined_df)

        print("Creating embeddings...")
        questions = processed_df["cleaned_question"].tolist()
        embeddings = self.create_embeddings(questions)

        print("Building FAISS index...")
        faiss_index = self.build_faiss_index(questions, embeddings)

        # Save processed data
        questions_mapping = {i: q for i, q in enumerate(questions)}
        # Write out to disk
        with open(
            f"{self.output_dir}/questions_mapping.json", "w", encoding="utf-8"
        ) as f:
            json.dump(questions_mapping, f, ensure_ascii=False, indent=2)

        return {
            "questions": questions,
            "embeddings": embeddings,
            "faiss_index": faiss_index,
            "questions_mapping": questions_mapping,
            "metadata": {
                "num_questions": len(questions),
                "embedding_dim": self.embedding_dim,
            },
        }
