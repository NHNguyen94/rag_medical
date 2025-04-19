from typing import List

import pandas as pd
import torch

from src.core_managers.encoding_manager import EncodingManager
from src.ml_models.lstm import LSTMModel
from src.utils.enums import LSTMConfig


class EmotionRecognitionService:
    def __init__(
            self,
            num_classes: int,
            input_dim: int = 1,
            hidden_dim: int = 10,
            layer_dim: int = 2
    ):
        self.lstm_config = LSTMConfig()
        self.encoder = EncodingManager()
        self.model = LSTMModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            layer_dim=layer_dim,
            output_dim=num_classes
        )

    def _reshape(self, data: torch.Tensor, max_length: int) -> torch.Tensor:
        return data.view(-1, max_length, 1)

    def prepare_data(self, data_path) -> (torch.Tensor, torch.Tensor):
        df = pd.read_csv(data_path)
        texts = df[self.lstm_config.TEXT_COL].tolist()
        labels = df[self.lstm_config.LABEL_COL].tolist()
        (tokenized_texts, max_length) = self.encoder.tokenize_texts(texts)
        padded_tokenized_texts = self.encoder.pad_sequences(tokenized_texts)
        X = self.encoder.to_tensor(padded_tokenized_texts, self.lstm_config.DTYPE_TEXT)
        y = self.encoder.to_tensor(labels, self.lstm_config.DTYPE_LABEL)

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
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        return self.model

    def predict(self, texts: List) -> torch.Tensor:
        (tokenized_texts, max_length) = self.encoder.tokenize_texts(texts)
        padded_tokenized_texts = self.encoder.pad_sequences(tokenized_texts)
        X = self.encoder.to_tensor(padded_tokenized_texts, self.lstm_config.DTYPE_TEXT)
        X = self._reshape(X, max_length)
        return self.model.predict_class(X)

    def evaluate_model(self, test_data_path: str) -> None:
        testX, testY = self.prepare_data(test_data_path)
        self.model.evaluate_model(testX, testY)


if __name__ == "__main__":
    emotion_recognition_service = EmotionRecognitionService(num_classes=6)
    emotion_recognition_service.train_model("src/data/emotion_data/test.csv", 3)
    emotion_recognition_service.load_model()
    predictions = emotion_recognition_service.predict(["I am happy", "I am sad"])
    print(f"Predictions: {predictions}")
    emotion_recognition_service.evaluate_model("src/data/emotion_data/test.csv")
