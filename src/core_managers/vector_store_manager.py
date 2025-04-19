from typing import Optional, List

import faiss
from llama_index.core.indices import VectorStoreIndex
from llama_index.core.indices.base import BaseIndex
from llama_index.core.node_parser.interface import TextSplitter
from llama_index.core.schema import Document, TransformComponent
from llama_index.core.service_context import ServiceContext
from llama_index.core.storage import StorageContext
from llama_index.vector_stores.faiss import FaissVectorStore


class VectorStoreManager:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.storage_context = self._initialize_storage()

    def _initialize_storage(
        self,
        dimension_for_embedding: Optional[int] = 1536,
    ) -> StorageContext:
        faiss_index = faiss.IndexFlatL2(dimension_for_embedding)
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        return storage_context

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
