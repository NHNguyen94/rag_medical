import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Dict

import pandas as pd
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from src.core_managers.encoding_manager import EncodingManager
from src.ml_models.cnn import CNNModel
from src.ml_models.lstm import LSTMModel
from src.utils.enums import EmotionRecognitionConfig
from src.utils.helpers import download_nlkt, clean_text


class EmotionRecognitionService:
    def __init__(
            self,
            # input_dim is to input data directly => float32
            # embedding_dim is to input token indices => long
            use_embedding: bool = False,
            embedding_dim: int = 100,
            hidden_dim: int = 10,
            layer_dim: int = 2,
            lr: float = 0.01,
            dropout: float = 0.2,
            num_classes: int = 6,
    ):
        self.encoder = EncodingManager()
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.use_embedding = use_embedding
        if self.use_embedding:
            self.input_dim = None
            self.lstm_model = LSTMModel(
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
            self.lstm_model = LSTMModel(
                input_dim=1,
                hidden_dim=hidden_dim,
                layer_dim=layer_dim,
                output_dim=num_classes,
                lr=lr,
                dropout=dropout,
            )
        self.cnn_model = CNNModel(
            output_dim=num_classes,
            lr=lr,
            vocab_size=len(self.encoder.tokenizer),
            padding_idx=self.encoder.tokenizer.pad_token_id,
            embedding_dim=embedding_dim,
            dropout=dropout,
        )
        self.config = EmotionRecognitionConfig()
        download_nlkt()

    def _reshape(self, data: torch.Tensor, max_length: int) -> torch.Tensor:
        if self.use_embedding:
            return data.view(-1, max_length)
        return data.view(-1, max_length, 1)

    def prepare_data(self, data_path: str) -> (torch.Tensor, torch.Tensor):
        df = pd.read_csv(data_path)
        texts = df[self.config.TEXT_COL].tolist()
        texts = [clean_text(text) for text in texts]
        labels = df[self.config.LABEL_COL].tolist()
        tokenized_texts = self.encoder.tokenize_texts(texts)
        # tokenized_texts = [self.encoder.tokenize_text(text) for text in texts]
        if self.use_embedding:
            X = self.encoder.to_tensor(tokenized_texts, self.config.LONG)
        else:
            X = self.encoder.to_tensor(tokenized_texts, self.config.FLOAT32)
        y = self.encoder.to_tensor(labels, self.config.LONG)

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

    def train_lstm_model(
            self,
            train_data_path: str,
            validation_data_path: str,
            num_epochs: int,
            model_path: str = None,
            batch_size: int = 32,
    ) -> None:
        train_X, train_y = self.prepare_data(train_data_path)
        val_X, val_y = self.prepare_data(validation_data_path)
        if model_path is None:
            model_path = self.config.LSTM_MODEL_PATH
        self.lstm_model.train_model(train_X, train_y, val_X, val_y, num_epochs, batch_size)
        model_config = {
            "num_classes": self.lstm_model.output_dim,
            "input_dim": self.lstm_model.input_dim,
            "hidden_dim": self.lstm_model.hidden_dim,
            "layer_dim": self.lstm_model.layer_dim,
            "lr": self.lstm_model.lr,
            "dropout": self.lstm_model.dropout,
            "vocab_size": len(self.encoder.tokenizer),
            "embedding_dim": self.lstm_model.embedding.embedding_dim,
        }
        torch.save(
            {
                "model_state_dict": self.lstm_model.state_dict(),
                "model_config": model_config,
            },
            model_path,
        )

    def train_cnn_model(
            self,
            train_data_path: str,
            validation_data_path: str,
            num_epochs: int,
            model_path: str = None,
            batch_size: int = 32,
    ) -> None:
        train_X, train_y = self.prepare_data(train_data_path)
        val_X, val_y = self.prepare_data(validation_data_path)
        if model_path is None:
            model_path = self.config.CNN_MODEL_PATH
        self.cnn_model.train_model(train_X, train_y, val_X, val_y, num_epochs, batch_size)
        model_config = {
            "num_classes": self.cnn_model.output_dim,
            "lr": self.cnn_model.lr,
            "dropout": self.cnn_model.dropout,
            "vocab_size": len(self.encoder.tokenizer),
            "embedding_dim": self.cnn_model.embedding.embedding_dim,
        }
        torch.save(
            {
                "model_state_dict": self.cnn_model.state_dict(),
                "model_config": model_config,
            },
            model_path,
        )

    def load_lstm_model(self, model_path: str = None) -> LSTMModel:
        if model_path is None:
            model_path = self.config.LSTM_MODEL_PATH
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
            embedding_dim=config["embedding_dim"],
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model

    def load_cnn_model(self, model_path: str = None) -> CNNModel:
        if model_path is None:
            model_path = self.config.CNN_MODEL_PATH
        checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
        config = checkpoint["model_config"]
        model = CNNModel(
            output_dim=config["num_classes"],
            lr=config["lr"],
            vocab_size=config["vocab_size"],
            embedding_dim=config["embedding_dim"],
            dropout=config["dropout"],
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model

    async def async_load_lstm_model(self, model_path: str = None) -> LSTMModel:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self.executor, self.load_lstm_model, model_path)
        return self.lstm_model

    async def async_load_cnn_model(self, model_path: str = None) -> CNNModel:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self.executor, self.load_cnn_model, model_path)
        return self.cnn_model

    def predict_by_lstm_model(self, text: str) -> torch.Tensor:
        text = clean_text(text)
        tokenized_text_list = self.encoder.tokenize_texts([text])
        tokenized_text = tokenized_text_list[0]
        if self.use_embedding:
            X = self.encoder.to_tensor(tokenized_text, self.config.LONG)
        else:
            X = self.encoder.to_tensor(tokenized_text, self.config.FLOAT32)
        X = X.to(next(self.lstm_model.parameters()).device)
        # X = self._reshape(X, max_length)
        return self.lstm_model.predict(X.unsqueeze(0))

    def predict_by_cnn_model(self, text: str) -> torch.Tensor:
        text = clean_text(text)
        tokenized_text_list = self.encoder.tokenize_texts([text])
        tokenized_text = tokenized_text_list[0]
        if self.use_embedding:
            X = self.encoder.to_tensor(tokenized_text, self.config.LONG)
        else:
            X = self.encoder.to_tensor(tokenized_text, self.config.FLOAT32)
        X = X.to(next(self.cnn_model.parameters()).device)
        # X = self._reshape(X, max_length)
        return self.cnn_model.predict(X.unsqueeze(0))

    def evaluate_lstm_model(self, test_data_path: str) -> Dict:
        df = pd.read_csv(test_data_path)
        texts = df[self.config.TEXT_COL].tolist()
        texts = [clean_text(text) for text in texts]
        labels = df[self.config.LABEL_COL].tolist()
        predictions = []
        for text in texts:
            prediction = self.predict_by_lstm_model(text)
            predictions.append(prediction.item())

        return {
            "model": "LSTM",
            "unique_predicted_labels": sorted(set(predictions)),
            "accuracy": 100 * accuracy_score(labels, predictions),
            "precision": 100 * precision_score(labels, predictions, average="weighted"),
            "recall": 100 * recall_score(labels, predictions, average="weighted"),
            "f1_score": 100 * f1_score(labels, predictions, average="weighted"),
        }

    def evaluate_cnn_model(self, test_data_path: str) -> Dict:
        df = pd.read_csv(test_data_path)
        texts = df[self.config.TEXT_COL].tolist()
        texts = [clean_text(text) for text in texts]
        labels = df[self.config.LABEL_COL].tolist()
        predictions = []
        for text in texts:
            prediction = self.predict_by_cnn_model(text)
            predictions.append(prediction.item())

        return {
            "model": "CNN",
            "unique_predicted_labels": sorted(set(predictions)),
            "accuracy": 100 * accuracy_score(labels, predictions),
            "precision": 100 * precision_score(labels, predictions, average="weighted"),
            "recall": 100 * recall_score(labels, predictions, average="weighted"),
            "f1_score": 100 * f1_score(labels, predictions, average="weighted"),
        }
