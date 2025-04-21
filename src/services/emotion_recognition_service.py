import asyncio
from typing import List

import pandas as pd
import torch
from concurrent.futures import ThreadPoolExecutor

from src.core_managers.encoding_manager import EncodingManager
from src.ml_models.lstm import LSTMModel
from src.utils.enums import LSTMConfig


class EmotionRecognitionService:
    def __init__(
        self,
        num_classes: int = 6,
        # input_dim is to input data directly => float32
        # embedding_dim is to input token indices => long
        use_embedding: bool = False,
        embedding_dim: int = 100,
        vocab_size: int = 10000,
        hidden_dim: int = 10,
        layer_dim: int = 2,
        lr: float = 0.01,
        dropout: float = 0.2,
    ):
        self.use_embedding = use_embedding
        if self.use_embedding:
            self.input_dim = None
            self.model = LSTMModel(
                hidden_dim=hidden_dim,
                layer_dim=layer_dim,
                output_dim=num_classes,
                lr=lr,
                vocab_size=vocab_size,
                embedding_dim=embedding_dim,
            )
        else:
            # Always 1 for input_dim
            self.model = LSTMModel(
                input_dim=1,
                hidden_dim=hidden_dim,
                layer_dim=layer_dim,
                output_dim=num_classes,
                lr=lr,
                dropout=dropout
            )
        self.lstm_config = LSTMConfig()
        self.encoder = EncodingManager()
        self.executor = ThreadPoolExecutor(max_workers=1)

    def _reshape(self, data: torch.Tensor, max_length: int) -> torch.Tensor:
        if self.use_embedding:
            return data.view(-1, max_length)
        return data.view(-1, max_length, 1)

    def prepare_data(self, data_path) -> (torch.Tensor, torch.Tensor):
        df = pd.read_csv(data_path)
        texts = df[self.lstm_config.TEXT_COL].tolist()
        labels = df[self.lstm_config.LABEL_COL].tolist()
        (tokenized_texts, max_length) = self.encoder.tokenize_texts(texts)
        padded_tokenized_texts = self.encoder.pad_sequences(tokenized_texts)
        if self.use_embedding:
            X = self.encoder.to_tensor(padded_tokenized_texts, self.lstm_config.LONG)
        else:
            X = self.encoder.to_tensor(padded_tokenized_texts, self.lstm_config.FLOAT32)
        y = self.encoder.to_tensor(labels, self.lstm_config.LONG)

        return self._reshape(X, max_length), y

    def train_model(
        self, train_data_path: str, num_epochs: int, model_path: str = None
    ) -> None:
        trainX, trainY = self.prepare_data(train_data_path)
        if model_path is None:
            model_path = self.lstm_config.MODEL_PATH
        self.model.train_model(trainX, trainY, num_epochs)
        torch.save(self.model.state_dict(), model_path)

    def load_model(self, model_path: str = None) -> LSTMModel:
        if model_path is None:
            model_path = self.lstm_config.MODEL_PATH
        with torch.no_grad():
            self.model.load_state_dict(
                torch.load(model_path, map_location=torch.device("cpu"))
            )
        self.model.eval()
        return self.model

    async def async_load_model(self, model_path: str = None) -> LSTMModel:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self.executor, self.load_model, model_path)
        return self.model

    def predict(self, texts: List) -> torch.Tensor:
        (tokenized_texts, max_length) = self.encoder.tokenize_texts(texts)
        padded_tokenized_texts = self.encoder.pad_sequences(tokenized_texts)
        if self.use_embedding:
            X = self.encoder.to_tensor(padded_tokenized_texts, self.lstm_config.LONG)
        else:
            X = self.encoder.to_tensor(padded_tokenized_texts, self.lstm_config.FLOAT32)
        X = self._reshape(X, max_length)
        return self.model.predict_class(X)

    def evaluate_model(self, test_data_path: str) -> None:
        testX, testY = self.prepare_data(test_data_path)
        self.model.evaluate_model(testX, testY)

