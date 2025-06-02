from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from pathlib import Path
import json
import torch
import faiss
from tqdm import tqdm

from src.core_managers.encoding_manager import EncodingManager
from src.core_managers.vector_store_manager import VectorStoreManager
from src.utils.helpers import clean_text
from src.utils.directory_manager import DirectoryManager


class QuestionDataProcessor:
    def __init__(
            self,
            data_dir: str = "../../data/fine_tune_dataset",
            output_dir: str = "../../data/processed",
            embedding_dim: int = 768
    ):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.embedding_dim = embedding_dim

        # Initialize components
        self.encoding_manager = EncodingManager()
        self.vector_store = VectorStoreManager()
        self.dir_manager = DirectoryManager()

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_and_combine_datasets(self) -> pd.DataFrame:
        """Load and combine all datasets from the fine_tune_dataset directory."""
        combined_data = []

        # Load all CSV files in the directory
        for file_path in self.data_dir.glob("*.csv"):
            df = pd.read_csv(file_path)
            # Add source information
            df['source'] = file_path.stem
            combined_data.append(df)

        # Combine all datasets
        combined_df = pd.concat(combined_data, ignore_index=True)

        return combined_df


    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df.head()
        df['cleaned_question'] = df['Question'].apply(clean_text)
        df = df.drop_duplicates(subset=['cleaned_question'])
        df = df.dropna(subset=['cleaned_question'])

        self.output_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.output_dir / "cleaned_dataset.csv", index=False)

        return df

    def create_embeddings(self, questions: List[str]) -> List[float]:
        """Create embeddings for questions using the encoding manager."""
        embeddings = []
        for question in tqdm(questions, desc="Creating embeddings"):
            # Tokenize and create embedding
            tokens = self.encoding_manager.tokenize_text(question)
            # Convert tokens to embedding
            embedding = self._tokens_to_embedding(tokens)
            embeddings.append(embedding)
        return embeddings

    def _tokens_to_embedding(self, tokens: List[int]) -> List[float]:
        """Convert tokens to embedding using the model."""
        tokens_tensor = self.encoding_manager.to_tensor([tokens], "long").to(self.encoding_manager.device)

        with torch.no_grad():
            embeddings = self.encoding_manager.model.embeddings(tokens_tensor)
            attention_mask = (tokens_tensor != self.encoding_manager.tokenizer.pad_token_id).float()
            mean_embedding = (embeddings * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1,
                                                                                                         keepdim=True)
            embedding = mean_embedding[0].cpu().numpy().tolist()

            if len(embedding) != self.embedding_dim:
                print(f"Mismatch detected: got {len(embedding)}, expected {self.embedding_dim}")
                if len(embedding) < self.embedding_dim:
                    embedding.extend([0.0] * (self.embedding_dim - len(embedding)))
                else:
                    embedding = embedding[:self.embedding_dim]

        return embedding

    def build_faiss_index(self, questions: List[str], embeddings: List[List[float]]):
        """Build FAISS index for question retrieval."""
        embeddings_array = np.array(embeddings).astype('float32')

        faiss_index = faiss.IndexFlatL2(self.embedding_dim)
        faiss_index.add(embeddings_array)

        return faiss_index

    def process_datasets(self) -> Dict:
        """Process all datasets and prepare for training."""
        print("Loading datasets...")
        combined_df = self.load_and_combine_datasets()

        print("Preprocessing data...")
        processed_df = self.preprocess_data(combined_df)

        print("Creating embeddings...")
        questions = processed_df['cleaned_question'].tolist()
        embeddings = self.create_embeddings(questions)
        embeddings_array = np.array(embeddings).astype('float32')

        print("Building FAISS index...")
        faiss_index = self.build_faiss_index(questions, embeddings)

        # Save FAISS index separately
        faiss_index_path = self.output_dir / "faiss_index.bin"
        faiss.write_index(faiss_index, str(faiss_index_path))

        # Save questions mapping
        questions_mapping = {i: q for i, q in enumerate(questions)}
        questions_mapping_path = self.output_dir / "questions_mapping.json"
        with open(questions_mapping_path, 'w') as f:
            json.dump(questions_mapping, f)

        # Save processed data
        processed_data = {
            'questions_mapping_path': str(questions_mapping_path),
            'faiss_index_path': str(faiss_index_path),
            'embeddings_path': str(self.output_dir / "embeddings.npy"),
            'metadata': {
                'num_questions': len(questions),
                'embedding_dim': self.embedding_dim
            }
        }

        np.save(processed_data['embeddings_path'], embeddings_array)

        # Save to file
        output_file = self.output_dir / "processed_data.json"
        with open(output_file, 'w') as f:
            json.dump(processed_data, f)

        return {
            'questions': questions,
            'embeddings': embeddings,
            'faiss_index': faiss_index,
            'questions_mapping': questions_mapping,
            'metadata': processed_data['metadata']
        }


