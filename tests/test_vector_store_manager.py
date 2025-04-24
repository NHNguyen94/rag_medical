import pytest
from unittest.mock import MagicMock, AsyncMock
from src.core_managers.vector_store_manager import VectorStoreManager
from llama_index.core.schema import Document
from llama_index.core.indices.base import BaseIndex
from llama_index.core.storage import StorageContext
from llama_index.vector_stores.faiss import FaissVectorStore
import pandas as pd
import numpy as np
import pickle
import hashlib
from openai import OpenAI

@pytest.fixture
def mock_sample_qa_dataset(mocker):
    mock_data = [
        {
            "question": "What are the symptoms of diabetes?",
            "answer": "Common symptoms include increased thirst, frequent urination, fatigue, and blurred vision.",
            "topic": "diabetes"
        },
        {
            "question": "What causes lung cancer?",
            "answer": "Lung cancer is primarily caused by smoking, but also by exposure to radon, asbestos, or air pollution.",
            "topic": "cancer"
        }
    ]
    mocker.patch("src.utils.helpers.sample_qa_data", return_value=mock_data)
    return mock_data

#Mock OpenAI client
@pytest.fixture
def mock_openai_client(mocker):
    mock_client = MagicMock(spec=OpenAI)
    mock_response = MagicMock()
    mock_response.data = [MagicMock(embedding=[0.1]*1536)]
    mock_client.embeddings.create = AsyncMock(return_value=mock_response)
    mocker.patch("openai.OpenAI", return_value = mock_client)
    return mock_client

#Mock FAISS and LlamaIndex dependencies
@pytest.fixture
def mock_faiss_and_llama_index(mocker):
    mocker.patch("faiss.IndexFlatL2", return_value=MagicMock())
    mocker.patch("faiss.IndexIVFFlat", return_value=MagicMock(train=MagicMock()))
    mocker.patch("llama_index.vector_stores.faiss.FaissVectorStore",  return_value=MagicMock(
        save_to_disk=MagicMock(),
        from_persist_dir=MagicMock()
    ))
    mocker.patch("llama_index.core.storage.StorageContext.from_defaults", return_value=MagicMock())
    mock_index = MagicMock(spec=BaseIndex)
    mock_query_engine = MagicMock()
    mock_response = MagicMock(response="Mock response")
    mock_response.source_nodes = [
        MagicMock(node=MagicMock(text="Magic Mock", metadata={ "question": "Mock Question", "answer": "Mock Answer", "topic": "Mock Topic" }), score=0.9),
    ]
    mock_query_engine.query = MagicMock(return_value=mock_response)
    mock_index.as_query_engine = MagicMock(return_value=mock_query_engine)
    mocker.patch("llama_index.core.indices.vector_store.VectorStoreIndex.from_documents", return_value=mock_index)
    mocker.patch("llama_index.core.indices.vector_store.VectorStoreIndex.from_vector_store", return_value=mock_index)
    return mock_index

@pytest.fixture
def vector_store_manager(mock_openai_client, mock_faiss_and_llama_index):
    return VectorStoreManager(user_id="test user", cache_path="test_embedding_cache.pkl")

def test_initialization(vector_store_manager):
    """ Test vector store manager initialization """
    assert vector_store_manager.user_id == "test user"
    assert vector_store_manager.cache_path == "test_embedding_cache.pkl"
    assert isinstance(vector_store_manager.embedding_cache, dict)
    assert vector_store_manager.storage_context is not None
    assert isinstance(vector_store_manager.client, MagicMock)


def test_load_embedding_cache_empty(vector_store_manager, mocker):
    """ Test loading an empty or non-existent embedding cache """
    mocker.patch("os.path.exists", return_value=False)
    cache = vector_store_manager.load_embedding_cache()
    assert cache == {}

def test_load_embedding_cache_existing(vector_store_manager, mocker):
    """ Test loading an existing embedding cache """
    cache_file = tmp_path / "test_embedding_cache.pkl"
    cache_data = { "hash1": [0.1] * 1536 }
    with open(cache_file, "wb") as f:
        pickle.dump(cache_data, f)
    vector_store_manager.cache_path = str(cache_file)
    cache = vector_store_manager.load_embedding_cache()
    assert cache == cache_data

def test_save_embedding_cache(vector_store_manager, tmp_path):
    """ Test saving an embedding cache """
    cache_file = tmp_path / "test_embedding_cache.pkl"
    vector_store_manager.cache_path = str(cache_file)
    vector_store_manager.embedding_cache = { "hash1": [0.1] * 1536 }
    vector_store_manager._save_embedding_cache()
    assert os.path.exists(cache_file)
    with open(cache_file, "rb") as f:
        saved_cache = pickle.load(f)
    assert saved_cache == { "hash1": [0.1] * 1536 }