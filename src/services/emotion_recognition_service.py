import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

import pandas as pd
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from src.core_managers.cnn_model_manager import CNNModelManager, CNNModel
from src.utils.enums import EmotionRecognitionConfig
from src.utils.helpers import clean_text, build_vocab

config = EmotionRecognitionConfig()


class EmotionRecognitionService:
    def __init__(
        self,
        train_data_path: str = config.TRAIN_DATA_PATH,
        test_data_path: str = config.TEST_DATA_PATH,
        validation_data_path: str = config.VALIDATION_DATA_PATH,
        embed_dim: int = config.DEFAULT_EMBED_DIM,
        num_classes: int = config.DEFAULT_NUM_CLASSES,
        kernel_sizes: List = config.DEFAULT_KERNEL_SIZES,
        num_filters: int = config.DEFAULT_NUM_FILTERS,
        dropout: float = config.DEFAULT_DROPOUT,
        lr: float = config.DEFAULT_LR,
    ):
        self.vocab = build_vocab(
            pd.read_csv(train_data_path)[config.TEXT_COL].tolist(),
        )
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.validation_data_path = validation_data_path
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.kernel_sizes = kernel_sizes
        self.num_filters = num_filters
        self.dropout = dropout
        self.model_manager = CNNModelManager(
            vocab_size=len(self.vocab),
            embed_dim=embed_dim,
            num_classes=num_classes,
            kernel_sizes=kernel_sizes,
            num_filters=num_filters,
            dropout=dropout,
            lr=lr,
        )
        self.min_freq = 1
        self.executor = ThreadPoolExecutor(max_workers=1)

    def train(
        self,
        batch_size: int = 32,
        epochs: int = 10,
    ) -> None:
        texts_train = pd.read_csv(self.train_data_path)[config.TEXT_COL].tolist()
        labels_train = pd.read_csv(self.train_data_path)[config.LABEL_COL].tolist()
        texts_train = [clean_text(text) for text in texts_train]
        val_texts = pd.read_csv(self.test_data_path)[config.TEXT_COL].tolist()
        val_labels = pd.read_csv(self.test_data_path)[config.LABEL_COL].tolist()
        val_texts = [clean_text(text) for text in val_texts]
        self.model_manager.train(
            texts_train=texts_train,
            labels_train=labels_train,
            texts_val=val_texts,
            labels_val=val_labels,
            vocab=self.vocab,
            max_len=config.MAX_SEQ_LENGTH,
            batch_size=batch_size,
            epochs=epochs,
            device=config.DEVICE,
        )

    def save_model(self, model_path: str = None) -> None:
        if model_path is None:
            model_path = config.CNN_MODEL_PATH
        torch.save(
            {
                "model_config": {
                    "vocab_size": len(self.vocab),
                    "embed_dim": self.embed_dim,
                    "num_classes": self.num_classes,
                    "kernel_sizes": self.kernel_sizes,
                    "num_filters": self.num_filters,
                    "dropout": self.dropout,
                },
                "model_state_dict": self.model_manager.model.state_dict(),
                "vocab": self.vocab,
            },
            model_path,
        )

    def load_model(self, model_path: str = None) -> (CNNModel, Dict[str, int]):
        if model_path is None:
            model_path = config.CNN_MODEL_PATH
        checkpoint = torch.load(model_path, map_location=config.DEVICE)

        model_config = checkpoint["model_config"]
        model = CNNModel(
            vocab_size=model_config["vocab_size"],
            embed_dim=model_config["embed_dim"],
            num_classes=model_config["num_classes"],
            kernel_sizes=model_config["kernel_sizes"],
            num_filters=model_config["num_filters"],
            dropout=model_config["dropout"],
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        vocab = checkpoint["vocab"]

        return model, vocab

    async def async_load_model(
        self, model_path: str = None
    ) -> (CNNModel, Dict[str, int]):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.load_model, model_path)

    def predict(
        self,
        text: str,
        model: CNNModel,
        vocab: Dict[str, int],
    ) -> torch.Tensor:
        encoded_text = self.model_manager.encode(text, vocab, config.MAX_SEQ_LENGTH)
        encoded_text = encoded_text.unsqueeze(0).to(config.DEVICE)
        model.to(config.DEVICE)
        model.eval()
        with torch.no_grad():
            output = model(encoded_text)
        return output.argmax(dim=1)

    def evaluate(
        self,
        model: CNNModel,
        vocab: Dict[str, int],
    ) -> Dict:
        df = pd.read_csv(self.validation_data_path)
        texts = df[config.TEXT_COL].tolist()
        texts = [clean_text(text) for text in texts]
        labels = df[config.LABEL_COL].tolist()
        predictions = []
        for text in texts:
            prediction = self.predict(
                text=text,
                model=model,
                vocab=vocab,
            )
            predictions.append(prediction.item())

        return {
            "unique_predicted_labels": sorted(set(predictions)),
            "accuracy": 100 * accuracy_score(labels, predictions),
            "precision": 100 * precision_score(labels, predictions, average="weighted"),
            "recall": 100 * recall_score(labels, predictions, average="weighted"),
            "f1_score": 100 * f1_score(labels, predictions, average="weighted"),
        }
