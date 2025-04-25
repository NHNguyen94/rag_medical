import os
import pytest
from unittest.mock import MagicMock, AsyncMock

from tomlkit import document

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
    mock_client = MagicMock()
    mock_embeddings = MagicMock()
    mock_response = MagicMock()
    mock_response.data = [MagicMock(embedding=[0.1]*1536)]
    mock_embeddings.create = AsyncMock(return_value=mock_response)
    mock_client.embeddings = mock_embeddings
    # Patch globally
    mocker.patch("openai.OpenAI", return_value=mock_client)
    # Patch locally in vector_store_manager
    mocker.patch("src.core_managers.vector_store_manager.OpenAI", return_value=mock_client)
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
    cache = vector_store_manager._load_embedding_cache()
    assert cache == {}

def test_load_embedding_cache_existing(vector_store_manager, mocker, tmp_path):
    """ Test loading an existing embedding cache """
    cache_file = tmp_path / "test_embedding_cache.pkl"
    cache_data = { "hash1": [0.1] * 1536 }
    with open(cache_file, "wb") as f:
        pickle.dump(cache_data, f)
    vector_store_manager.cache_path = str(cache_file)
    cache = vector_store_manager._load_embedding_cache()
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

def test_get_embedding_cache_hit(vector_store_manager, mocker):
    """ Test retrieving an embedding from cache """
    text = "Test text"
    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    embedding = [0.1] * 1536
    vector_store_manager.embedding_cache = { text_hash: embedding }
    result = vector_store_manager._get_embedding(text)
    assert result == embedding
    assert not vector_store_manager.client.embeddings.create.called

def test_get_embedding_cache_miss(vector_store_manager, mock_openai_client):
    """ Test generating an embedding from on cache miss """
    text = "Test text"
    embedding = [0.1] * 1536
    result = vector_store_manager._get_embedding(text)
    assert result == embedding
    mock_openai_client.embeddings.create.assert_called_once()
    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert vector_store_manager.embedding_cache[text_hash] == embedding
    # Verify cache was saved
    with open(vector_store_manager.cache_path, "rb") as f:
        saved_cache = pickle.load(f)
    assert saved_cache[text_hash] == embedding

def test_get_embedding_api_error(vector_store_manager, mock_openai_client):
    """ Test handling OpenAI API errors """
    text = "Test text"
    mock_openai_client.embeddings.create.side_effect = Exception("Mock Exception")
    result = vector_store_manager._get_embedding(text)
    assert result == []
    mock_openai_client.embeddings.create.assert_called_once()

def test_load_and_preprocess_data(vector_store_manager, mock_sample_qa_dataset):
    """ Test loading and preprocessing sample data """
    print(f"Mocked sample_qa_data: {mock_sample_qa_dataset}, {len(mock_sample_qa_dataset)}")
    documents = vector_store_manager.load_and_preprocess_data(file_path="dummy_path")
    print(f"Documents: {len(documents)}")
    assert len(documents) == 2
    assert all(isinstance(doc, Document) for doc in documents)
    assert documents[0].text == ("What are the symptoms of diabetes? Common symptoms include increased thirst "
                                 "frequent urination, fatigue, and blurred vision.")
    assert documents[0].metadata == {
        "question": "What are the symptoms of diabetes?",
        "answer": "Common symptoms include increased thirst, frequent urination, fatigue, and blurred vision.",
        "topic": "diabetes"
    }

def test_load_and_preprocess_data_empty(vector_store_manager, mocker):
    """ Test handling empty or invalid sample data """
    mocker.patch("src.utils.helpers.sample_qa_data", return_value=[])
    documents = vector_store_manager.load_and_preprocess_data(file_path="dummy_path")
    assert documents == []

def test_build_index(vector_store_manager, mock_sample_qa_dataset, mock_faiss_and_llama_index):
    """ Test building a FAISS index """
    documents = vector_store_manager.load_and_preprocess_data(file_path="dummy_path")
    index = vector_store_manager.build_index(documents, show_progress=True, batch_size=1)
    assert index is not None
    assert len(documents) == 2
    assert all(doc.embedding is not None for doc in documents)
    assert mock_faiss_and_llama_index.from_documents.called

def test_build_index_embedding_failure(vector_store_manager, mock_sample_qa_dataset, mocker):
    """ Test handling embedding failure during index building. """
    mocker.patch.object(vector_store_manager, "_get_embedding", return_value=[])
    documents = vector_store_manager.load_and_preprocess_data(file_path="dummy_path")
    with pytest.raises(ValueError, match="Failed to embed text"):
        vector_store_manager.build_index(documents, show_progress=True, batch_size=1)

def test_query_index(vector_store_manager, mock_sample_qa_dataset, mock_faiss_and_llama_index):
    """ Test querying the index without topic filter """
    documents = vector_store_manager.load_and_preprocess_data(file_path="dummy_path")
    index = vector_store_manager.build_index(documents, show_progress=True)
    result = vector_store_manager.query_index(index, query="What are diabetes symptoms?", top_k=1)
    assert result["response"] == "Mock response"
    assert len(result["sources"]) == 1
    assert result["sources"][0]["text"] == "Mock text"
    assert result["sources"][0]["score"] == 0.9

def test_query_index_with_topic_filter(vector_store_manager, mock_sample_qa_dataset, mock_faiss_and_llama_index):
    """ Test querying the index with topic filter """
    documents = vector_store_manager.load_and_preprocess_data(file_path="dummy_path")
    index = vector_store_manager.build_index(documents, show_progress=True)
    result = vector_store_manager.query_index(index, query="What are diabetes symptoms?", top_k=1, topic_filter="diabetes")
    assert result["response"] == "Mock response"
    assert len(result["sources"]) == 1
    assert result["sources"][0]["metadata"]["topic"] == "diabetes"

def test_query_index_error(vector_store_manager, mock_sample_qa_dataset, mock_faiss_and_llama_index):
    """ Test handling query errors """
    documents = vector_store_manager.load_and_preprocess_data(file_path="dummy_path")
    index = vector_store_manager.build_index(documents, show_progress=True)
    index.as_query_engine.side_effect = Exception("Query error")
    result = vector_store_manager.query_index(index, query="Test query", top_k=1)
    assert result == { "response": None, "sources": [] }

def test_save_index(vector_store_manager, mock_sample_qa_dataset, mock_faiss_and_llama_index, tmp_path):
        """ Test saving the index """
        documents = vector_store_manager.load_and_preprocess_data(file_path="dummy_path")
        index = vector_store_manager.build_index(documents, show_progress=True)
        index_path = str(tmp_path / "faiss_index")
        vector_store_manager.save_index(index, path=index_path)
        assert index.storage_context.vector_store.save_to_disk.called

def test_load_index(vector_store_manager, mock_faiss_and_llama_index, tmp_path):
    """  Test loading the index"""
    index_path = str(tmp_path / "faiss_index")
    index = vector_store_manager.load_index(path=index_path)
    assert index is not None
    assert mock_faiss_and_llama_index.from_vector_store.called

def test_load_index_error(vector_store_manager, mocker):
    """ Test handling load index errors """
    mocker.patch("llama_index.vector_stores.faiss.FaissVectorStore.from_persist_dir", side_effect=Exception("Load error"))
    index = vector_store_manager.load_index(path="invalid_path")
    assert index is None
