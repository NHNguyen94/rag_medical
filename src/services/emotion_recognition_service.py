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
        hidden_dim: int = 10,
        layer_dim: int = 2,
        lr: float = 0.01,
        dropout: float = 0.2,
    ):
        self.encoder = EncodingManager()
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.use_embedding = use_embedding
        if self.use_embedding:
            self.input_dim = None
            self.model = LSTMModel(
                hidden_dim=hidden_dim,
                layer_dim=layer_dim,
                output_dim=num_classes,
                lr=lr,
                vocab_size=len(self.encoder.tokenizer),
                padding_idx=self.encoder.tokenizer.pad_token_id,
                embedding_dim=embedding_dim,
                dropout=dropout,
            )
        else:
            # Always 1 for input_dim
            self.model = LSTMModel(
                input_dim=1,
                hidden_dim=hidden_dim,
                layer_dim=layer_dim,
                output_dim=num_classes,
                lr=lr,
                dropout=dropout,
            )
        self.lstm_config = LSTMConfig()

    def _reshape(self, data: torch.Tensor, max_length: int) -> torch.Tensor:
        if self.use_embedding:
            return data.view(-1, max_length)
        return data.view(-1, max_length, 1)

    def prepare_data(self, data_path: str) -> (torch.Tensor, torch.Tensor):
        df = pd.read_csv(data_path)
        texts = df[self.lstm_config.TEXT_COL].tolist()
        labels = df[self.lstm_config.LABEL_COL].tolist()
        (tokenized_texts, max_length) = self.encoder.tokenize_texts(texts)
        if self.use_embedding:
            X = self.encoder.to_tensor(tokenized_texts, self.lstm_config.LONG)
        else:
            X = self.encoder.to_tensor(tokenized_texts, self.lstm_config.FLOAT32)
        y = self.encoder.to_tensor(labels, self.lstm_config.LONG)

        # print("Shape X:", X.shape)
        # print("Shape y:", y.shape)
        assert X.shape[0] == y.shape[0], "Mismatch"

        # Use reshape if don't fix max length
        # new_X, new_y = self._reshape(X, max_length), y
        #
        # print("Shape new_X:", new_X.shape)
        # print("Shape new_y:", new_y.shape)
        # assert new_X.shape[0] == new_y.shape[0], "Mismatch after reshape!"

        return X, y

    def train_model(
        self,
        train_data_path: str,
        num_epochs: int,
        model_path: str = None,
        batch_size: int = 32,
    ) -> None:
        trainX, trainY = self.prepare_data(train_data_path)
        if model_path is None:
            model_path = self.lstm_config.MODEL_PATH
        self.model.train_model(trainX, trainY, num_epochs, batch_size)
        model_config = {
            "num_classes": self.model.output_dim,
            "input_dim": self.model.input_dim,
            "hidden_dim": self.model.hidden_dim,
            "layer_dim": self.model.layer_dim,
            "lr": self.model.lr,
            "dropout": self.model.dropout,
            "vocab_size": len(self.encoder.tokenizer),
            "embedding_dim": self.model.embedding.embedding_dim,
        }
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "model_config": model_config,
            },
            model_path,
        )

    def load_model(self, model_path: str = None) -> LSTMModel:
        if model_path is None:
            model_path = self.lstm_config.MODEL_PATH
        checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
        config = checkpoint["model_config"]
        model = LSTMModel(
            input_dim=config["input_dim"],
            hidden_dim=config["hidden_dim"],
            layer_dim=config["layer_dim"],
            output_dim=config["num_classes"],
            lr=config["lr"],
            dropout=config["dropout"],
            vocab_size=config["vocab_size"],
            embedding_dim=config["embedding_dim"]
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model

    async def async_load_model(self, model_path: str = None) -> LSTMModel:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self.executor, self.load_model, model_path)
        return self.model

    def predict(self, texts: List) -> torch.Tensor:
        (tokenized_texts, max_length) = self.encoder.tokenize_texts(texts)
        if self.use_embedding:
            X = self.encoder.to_tensor(tokenized_texts, self.lstm_config.LONG)
        else:
            X = self.encoder.to_tensor(tokenized_texts, self.lstm_config.FLOAT32)
        # X = self._reshape(X, max_length)
        return self.model.predict_class(X)

    def evaluate_model(self, test_data_path: str) -> None:
        testX, testY = self.prepare_data(test_data_path)
        self.model.evaluate_model(testX, testY)
