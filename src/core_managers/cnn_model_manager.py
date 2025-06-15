from typing import List, Dict

import pandas as pd
import torch
from torch import Tensor
from torch.nn import (
    Conv2d,
    Embedding,
    ModuleList,
    Dropout,
    Linear,
    Module,
    CrossEntropyLoss,
)
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from src.utils.helpers import build_vocab, clean_and_tokenize


class CNNModel(Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_classes: int,
        kernel_sizes: List = [3, 4, 5],
        num_filters: int = 100,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.embedding = Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = ModuleList(
            [Conv2d(1, num_filters, (k, embed_dim)) for k in kernel_sizes]
        )
        self.dropout_layer = Dropout(dropout)
        self.fc = Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x)
        x = x.unsqueeze(1)
        conv_x = [torch.relu(conv(x)).squeeze(3) for conv in self.convs]
        pooled_x = [torch.max(c, dim=2)[0] for c in conv_x]
        cat_x = torch.cat(pooled_x, dim=1)
        out = self.dropout_layer(cat_x)
        return self.fc(out)


class CNNModelManager:
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_classes: int,
        lr: float,
        kernel_sizes: List,
        num_filters: int,
        dropout: float,
    ):
        self.min_freq = 1
        self.model = CNNModel(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_classes=num_classes,
            kernel_sizes=kernel_sizes,
            num_filters=num_filters,
            dropout=dropout,
        )
        self.criterion = CrossEntropyLoss()
        self.optimizer = Adam(self.model.parameters(), lr=lr)

    def encode(self, text: str, vocab: Dict[str, int], max_len: int) -> Tensor:
        tokens = clean_and_tokenize(text)
        idxs = [vocab.get(t, vocab["<unk>"]) for t in tokens]
        if len(idxs) < max_len:
            idxs = idxs + [vocab["<pad>"]] * (max_len - len(idxs))
        else:
            idxs = idxs[:max_len]
        return torch.tensor(idxs)

    def create_dataset(
        self, texts: List[str], labels: List[int], vocab: Dict[str, int], max_len: int
    ) -> TensorDataset:
        encoded_texts = [self.encode(text, vocab, max_len) for text in texts]
        label_tensors = [torch.tensor(label) for label in labels]
        return TensorDataset(torch.stack(encoded_texts), torch.stack(label_tensors))

    def train(
        self,
        texts_train: List[str],
        labels_train: List[int],
        texts_val: List[str],
        labels_val: List[int],
        vocab: Dict[str, int],
        max_len: int,
        batch_size: int,
        epochs: int,
        device: torch.device,
    ) -> (float, float):
        dataset = self.create_dataset(texts_train, labels_train, vocab, max_len)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        val_dataset = self.create_dataset(texts_val, labels_val, vocab, max_len)
        val_data_loader = DataLoader(val_dataset, batch_size=batch_size)

        self.model.to(device)
        self.model.train()

        for epoch in range(epochs):
            total_loss = 0.0
            for batch in data_loader:
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(data_loader)}")

            final_train_loss = total_loss / len(data_loader)

            self.model.eval()
            with torch.no_grad():
                val_loss = 0.0
                correct = 0
                total = 0

                for batch in val_data_loader:
                    inputs, targets = batch
                    inputs, targets = inputs.to(device), targets.to(device)

                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                    val_loss += loss.item()

                    _, predicted = torch.max(outputs.data, 1)
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()

                print(
                    f"Validation Loss: {val_loss / len(val_data_loader)}, Accuracy: {correct / total:.4f}"
                )

            final_val_loss = val_loss / len(val_data_loader)

            self.model.train()

        return final_train_loss, final_val_loss
