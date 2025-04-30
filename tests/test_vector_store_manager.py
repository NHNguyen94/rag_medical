from llama_index.core.indices.base import BaseIndex

from src.core_managers.vector_store_manager import VectorStoreManager


class TestVectorStore:
    vt_store = VectorStoreManager(user_id="test_user")

    def test_build_or_load_index(self):
        index = self.vt_store.build_or_load_index(
            storage_path="tests/resources/indices",
        )
        assert isinstance(index, BaseIndex)
