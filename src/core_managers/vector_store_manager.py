import hashlib
import pickle
from http.client import responses
from typing import Optional, List, Dict

import os
import faiss
import dotenv
import numpy as np
import pandas as pd
from llama_index.core.base.embeddings.base import similarity
from llama_index.core.indices import VectorStoreIndex
from llama_index.core.indices.base import BaseIndex
from llama_index.core.node_parser.interface import TextSplitter
from llama_index.core.schema import Document, TransformComponent
from llama_index.core.service_context import ServiceContext
from llama_index.core.storage import StorageContext
from llama_index.vector_stores.faiss import FaissVectorStore
from openai import OpenAI
from sqlalchemy.engine import row

from src.utils.helpers import sample_qa_data

dotenv.load_dotenv()

class VectorStoreManager:
    def __init__(self, user_id: str, cache_path: str = "embedding_cache.pkl"):
        self.user_id = user_id
        self.cache_path = cache_path
        self.storage_context = self._initialize_storage()
        self.embedding_cache =  self._load_embedding_cache()
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def _load_embedding_cache(self) -> Dict[str, List[float]]:
        """
        Load the embedding from the cache path
        :return:
        """
        try:
            if os.path.exists(self.cache_path):
                with open(self.cache_path, "rb") as f:
                    return pickle.load(f)
            return {}
        except Exception as e:
            print(f"Failed to load embedding cache: {e}")
            return {}

    def _save_embedding_cache(self) -> None:
        """
        Save the embedding to the cache path
        """
        try:
            with open(self.cache_path, "wb") as f:
                pickle.dump(self.embedding_cache, f)
        except Exception as e:
            print(f"Failed to save embedding cache: {e}")

    def _initialize_storage(
        self,
        dimension_for_embedding: Optional[int] = 1536,
    ) -> StorageContext:
        """
        Initializes FAISS vector store with IndexIVFFlat for scalability
        :param dimension_for_embedding:
        :return StrorageContext: LlamaIndex storage context with FAISS vector store.
        """
        quantizer = faiss.IndexFlatL2(dimension_for_embedding)
        faiss_index = faiss.IndexIVFFlat(quantizer, dimension_for_embedding, nlist=100)
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        return storage_context

    def _get_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text using OpenAI's text-embedding-ada-002.
        :param text: Text to embed.
        :return List[float]: Embedding vector.
        """

        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

        #Check cache
        if text_hash in self.embedding_cache:
            return self.embedding_cache[text_hash]
        try:
            response = self.client.embeddings.create(
                input=text,
                model="text-embedding-ada-002"
            )
            embedding = response.data[0].embedding
            # Store in cache
            self.embedding_cache[text_hash] = embedding
            # Save cache immediately
            self._save_embedding_cache()
            return embedding
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return []

    def load_and_preprocess_data(self, file_path:str) -> List[Document]:
        """
        Load the dataset and convert to LlamaIndex documents.
        :param file_path: Path to dataset.
        :return List[Document]: LlamaIndex documents.
        """
        try:
            data = sample_qa_data()
            df = pd.DataFrame()
            df["text"] = data["question"] + " " + data["answer"]

            # Convert to Documents
            documents = [
                Document(
                    text=row["text"],
                    metadata={
                        "question": row["question"],
                        "answer": row["answer"],
                        "topic": row["topic"]
                    }
                )
                for _, row in df.iterrows()
            ]
            return documents
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return []


    def build_index(
        self,
        documents: List[Document],
        show_progress: bool = False,
        batch_size: int = 100
    ) -> BaseIndex:
        """
        Build a vector store index from documents using OpenAI's text-embedding-ada-002.
        :param documents: List of document objects.
        :param show_progress:
        :param batch_size: Batch size for embedding to manage API rate limiting.
        :return: BaseIndex: LlamaIndex vector store index.
        """
        try:
            # Embed doucments in batches
            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i : i + batch_size]
                texts = [doc.text for doc in batch_docs]
                embeddings = []
                for text in texts:
                    embedding = self._get_embedding(text)
                    if embedding:
                        embeddings.append(embedding)
                    else:
                        raise ValueError(f"Failed to embed text: {text}")
                for doc, embedding in zip(batch_docs, embeddings):
                    doc.embedding = embedding

            # Train FAISS index
            embeddings = np.array([doc.embedding for doc in documents], dtype=np.float32)
            self.storage_context.vector_store.faiss_index.train(embeddings)

            index = VectorStoreIndex.from_documents(
                documents=documents,
                storage_context=self.storage_context,
                show_progress=show_progress,
            )
            return index
        except Exception as e:
            print(f"Error building index: {e}")
            return None

    def query_index(
            self,
            index: BaseIndex,
            query: str,
            top_k: int = 5,
            topic_filter: Optional[str] = None
    ) -> Dict:

        try:
            query_engine = index.as_query_engine(similarity_top_k=top_k)
            if topic_filter:
                query_engine.filters = { 'topic': topic_filter }

            response = query_engine.query(query)
            return {
                "response": response.response,
                "sources": [
                    {
                        "text": node.node.text,
                        "metadata": node.node.metadata,
                        "score": node.score
                    }
                    for node in response.source_nodes
                ]
            }
        except Exception as e:
            print(f"Error querying index: {e}")
            return { "response": None, "sources": [] }


    def save_index(self, index: BaseIndex, path: str="faiss_index") -> None:
        """
        Save index to disk.
        :param index: LlamaIndex index.
        :param path: Path to save index.
        :return: none
        """

        try:
            index.storage_context.vector_store.save_to_disk(path)
        except Exception as e:
            print(f"Error saving index: {e}")

    def load_index(self, path: str="faiss_index") -> BaseIndex:
        """
        Load index from disk.
        :param path: Directory to load index from.
        :return: Loaded LlamaIndex index.
        """

        try:
            vector_store = FaissVectorStore.from_persist_dir(path)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)
            return index
        except Exception as e:
            print(f"Error loading index: {e}")
            return None


