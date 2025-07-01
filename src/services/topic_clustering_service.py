import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List

import pandas as pd
import torch

from src.core_managers.encoding_manager import EncodingManager
from src.core_managers.transformer_model_manager import TransformerModelManager, TransformerModel
from src.utils.enums import TopicClusteringConfig

topic_config = TopicClusteringConfig()


class TopicClusteringService:
    def __init__(
            self,
            model_name: str = topic_config.DEFAULT_MODEL,
            train_data_path: str = topic_config.TRAIN_DATA_PATH,
            test_data_path: str = topic_config.TEST_DATA_PATH,
            validation_data_path: str = topic_config.TEST_DATA_PATH,
            dropout: float = topic_config.DEFAULT_DROPOUT,
            lr: float = topic_config.DEFAULT_LR,
            num_classes: int = topic_config.DEFAULT_NUM_CLASSES,
    ):
        self.model_name = model_name
        self.num_classes = num_classes
        self.dropout = dropout
        self.lr = lr
        self.encoder = EncodingManager()
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.model_manager = TransformerModelManager(
            model_name=model_name,
            num_classes=num_classes,
            lr=lr,
        )

    def train(
            self,
            batch_size: int = 32,
            epochs: int = 10,
    ) -> (float, float):
        texts_train = pd.read_csv(self.train_data_path)[
            topic_config.TEXT_COL
        ].tolist()
        labels_train = pd.read_csv(self.train_data_path)[
            topic_config.LABEL_COL
        ].tolist()
        texts_val = pd.read_csv(self.test_data_path)[topic_config.TEXT_COL].tolist()
        labels_val = pd.read_csv(self.test_data_path)[topic_config.LABEL_COL].tolist()

        (train_loss, val_loss) = self.model_manager.train(
            epochs=epochs,
            texts_train=texts_train,
            labels_train=labels_train,
            texts_val=texts_val,
            labels_val=labels_val,
            batch_size=batch_size,
            max_len=topic_config.MAX_SEQ_LENGTH,
            device=topic_config.DEVICE
        )

        return train_loss, val_loss

    def evaluate(self, texts: List[str], labels: List[int]) -> float:
        encoded_texts = self.model_manager.tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=topic_config.MAX_SEQ_LENGTH,
            return_tensors="pt"
        )
        input_ids = encoded_texts["input_ids"].to(topic_config.DEVICE)
        attention_mask = encoded_texts["attention_mask"].to(topic_config.DEVICE)

        with torch.no_grad():
            outputs = self.model_manager.model(input_ids, attention_mask)
            predictions = torch.argmax(outputs, dim=1)

        correct_predictions = (predictions.cpu().numpy() == labels).sum()
        accuracy = correct_predictions / len(labels)

        return accuracy

    def save_model(self, model_path: str = None) -> None:
        if model_path is None:
            model_path = topic_config.MODEL_PATH
        torch.save(
            {
                "model_config": {
                    "model_name": self.model_name,
                    "num_classes": self.num_classes,
                    "dropout": self.dropout,
                    "lr": self.lr,
                },
                "model_state_dict": self.model_manager.model.state_dict(),
            },
            model_path
        )

    def load_model(self, model_path: str = None) -> TransformerModel:
        if model_path is None:
            model_path = topic_config.MODEL_PATH
        checkpoint = torch.load(model_path, map_location=topic_config.DEVICE)

        model_config = checkpoint["model_config"]
        model = TransformerModel(
            model_name=model_config["model_name"],
            num_classes=model_config["num_classes"],
        )
        model.load_state_dict(checkpoint["model_state_dict"])

        model.eval()

        return model

    async def async_load_model(
            self, model_path: str = None
    ) -> TransformerModel:
        if model_path is None:
            model_path = topic_config.MODEL_PATH
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.load_model, model_path)

    def predict(self, text: str, model: TransformerModel) -> int:
        encoded_text = self.model_manager.tokenizer(
            [text],
            truncation=True,
            padding="max_length",
            max_length=topic_config.MAX_SEQ_LENGTH,
            return_tensors="pt"
        )
        input_ids = encoded_text["input_ids"].to(topic_config.DEVICE)
        attention_mask = encoded_text["attention_mask"].to(topic_config.DEVICE)
        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
            predicted = torch.argmax(outputs, dim=1)

        return predicted.cpu().numpy().tolist()[0]


if __name__ == "__main__":
    topic_clustering_service = TopicClusteringService()
    train_loss, val_loss = topic_clustering_service.train(
        batch_size=32,
        epochs=1
    )
    print(f"Training Loss: {train_loss}, Validation Loss: {val_loss}")
    topic_clustering_service.save_model()

    model = topic_clustering_service.load_model()

    texts = [
        "What are the symptoms of diabetes?",
        "How can I manage my hypertension?",
        "What is the best treatment for anxiety disorders?",
        "What are the side effects of chemotherapy?",
        "How can I improve my mental health?"
    ]

    for text in texts:
        predicted_topic = topic_clustering_service.predict(text, model)
        print(f"Text: {text}, Predicted Topic: {predicted_topic}")
