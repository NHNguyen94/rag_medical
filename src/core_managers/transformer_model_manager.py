from typing import Dict, List

import torch
from sklearn.preprocessing import LabelEncoder
from torch import Tensor
from torch.nn import Module, Linear, CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel
from transformers import AutoTokenizer


class TransformerTextDataset(Dataset):
    def __init__(
            self,
            texts: List[str],
            labels: List[int],
            tokenizer: AutoTokenizer,
            max_len: int
    ):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt"
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict:
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "label": self.labels[idx]
        }


class TransformerModel(Module):
    def __init__(self, model_name: str, num_classes: int):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.transformer.config.hidden_size
        self.classifier = Linear(self.hidden_size, num_classes)

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)

        return logits


class TransformerModelManager:
    def __init__(
            self,
            model_name: str,
            num_classes: int,
            lr: float,
    ):
        self.model = TransformerModel(model_name, num_classes)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.label_encoder = LabelEncoder()
        self.criterion = CrossEntropyLoss()
        self.optimizer = Adam(self.model.parameters(), lr=lr)
        self.min_freq = 1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_data_loaders(
            self,
            texts_train: List[str],
            labels_train: List[int],
            texts_val: List[str],
            labels_val: List[int],
            max_len: int,
            batch_size: int
    ) -> (DataLoader, DataLoader):
        texts_train = [str(text) for text in texts_train]
        texts_val = [str(text) for text in texts_val]
        train_dataset = TransformerTextDataset(texts_train, labels_train, self.tokenizer, max_len=max_len)
        test_dataset = TransformerTextDataset(texts_val, labels_val, self.tokenizer, max_len=max_len)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        return train_loader, test_loader

    def train(
            self, epochs: int,
            texts_train: List[str],
            labels_train: List[int],
            texts_val: List[str],
            labels_val: List[int],
            batch_size: int,
            max_len: int,
            device: torch.device
    ) -> (float, float):
        train_loader, val_loader = self.get_data_loaders(
            texts_train,
            labels_train,
            texts_val,
            labels_val,
            max_len=max_len,
            batch_size=batch_size
        )
        self.model.to(device)
        self.model.train()

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            correct = 0
            total = 0

            for batch in train_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                self.optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item() * input_ids.size(0)
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / total:.4f}, Accuracy: {correct / total:.4f}")
            final_train_loss = total_loss / total

            self.model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["label"].to(device)

                    outputs = self.model(input_ids, attention_mask)
                    loss = self.criterion(outputs, labels)

                    val_loss += loss.item() * input_ids.size(0)
                    _, predicted = outputs.max(1)
                    val_correct += predicted.eq(labels).sum().item()
                    val_total += labels.size(0)

            print(f"Validation Loss: {val_loss / val_total:.4f}, Accuracy: {val_correct / val_total:.4f}")
            final_val_loss = val_loss / val_total

            self.model.train()

        return final_train_loss, final_val_loss

