from typing import Optional, List

import os
import faiss
import dotenv
from llama_index.core.indices import VectorStoreIndex
from llama_index.core.indices.base import BaseIndex
from llama_index.core.node_parser.interface import TextSplitter
from llama_index.core.schema import Document, TransformComponent
from llama_index.core.service_context import ServiceContext
from llama_index.core.storage import StorageContext
from llama_index.vector_stores.faiss import FaissVectorStore
from openai import OpenAI

dotenv.load_dotenv()

class VectorStoreManager:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.storage_context = self._initialize_storage()
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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
        try:
            response = self.client.embeddings.create(
                input=text,
                model="text-embedding-ada-002"
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return []

    def load_and_preprocess_data(self, file_path:str) -> List[Document]:
        """
        Load the dataset and convert to LlamaIndex documents.
        :param file_path: Path to dataset.
        :return List[Document]: LlamaIndex documents.
        """
        pass

    def build_index(
        self,
        documents: List[Document],
        show_progress: bool = False,
    ) -> BaseIndex:
        index = VectorStoreIndex.from_documents(
            documents=documents,
            storage_context=self.storage_context,
            show_progress=show_progress,
        )
        return index
