from typing import Optional, List

import faiss
from llama_index.core.indices import load_index_from_storage, VectorStoreIndex
from llama_index.core.indices.base import BaseIndex
from llama_index.core.node_parser.interface import TextSplitter
from llama_index.core.schema import Document, TransformComponent
from llama_index.core.storage import StorageContext
from llama_index.vector_stores.faiss import FaissVectorStore

from src.utils.directory_manager import DirectoryManager


class VectorStoreManager:
    def __init__(self):
        pass

    def _initialize_storage(
        self,
        dimension_for_embedding: Optional[int] = 1536,
    ) -> StorageContext:
        """Initialize index with documents."""
        faiss_index = faiss.IndexFlatL2(dimension_for_embedding)
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        return storage_context

    def build_index(
        self,
        documents: List[Document],
        storage_context: StorageContext,
        show_progress: bool = False,
    ) -> BaseIndex:
        index = VectorStoreIndex.from_documents(
            documents=documents,
            storage_context=storage_context,
            show_progress=show_progress,
        )
        return index

    def store_index(self, index: BaseIndex, storage_path: str) -> None:
        """Store index to disk."""
        index.storage_context.persist(persist_dir=storage_path)

    def load_index(self, storage_path: str) -> BaseIndex:
        """Load index from disk."""
        vector_store = FaissVectorStore.from_persist_dir(storage_path)
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store,
            persist_dir=storage_path,
        )
        index = load_index_from_storage(storage_context=storage_context)
        return index

    def build_or_load_index(
        self,
        storage_path: str,
        documents: Optional[List[Document]] = None,
        # Later might need to use text_splitter and transformations
        text_splitter: Optional[TextSplitter] = None,
        transformations: Optional[List[TransformComponent]] = None,
    ) -> BaseIndex:
        if documents is None:
            documents = []
        if DirectoryManager.check_if_dir_exists(storage_path):
            return self.load_index(storage_path)
        else:
            index = self.build_index(
                documents=documents,
                storage_context=self._initialize_storage(),
            )
            self.store_index(index, storage_path)
            return index

    def append_document(
        self,
        storage_path: str,
        document: Document,
    ) -> None:
        if DirectoryManager.check_if_dir_exists(storage_path):
            index = self.load_index(storage_path)
            index.insert(document)
            self.store_index(index, storage_path)
        else:
            index = self.build_index(
                documents=[document],
                storage_context=self._initialize_storage(),
            )
            self.store_index(index, storage_path)
