import torch
from torch.nn import (
    Module,
    Linear,
    Embedding,
    CrossEntropyLoss,
    ModuleList,
    Dropout,
    Conv2d
)
from torch.optim import AdamW, Adam
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.functional import relu
from collections import Counter
from typing import List, Tuple

class CNNModel(Module):
    def __init__(
            self,
            output_dim: int,
            lr: float,
            kernel_sizes: List = [3, 4, 5],
            num_filters: int = 100,
            vocab_size: int = 10000,
            embedding_dim: int = 100,
            dropout: float = 0.2,
            padding_idx: int = 0,
    ):
        super().__init__()
        self.embedding = Embedding(
            vocab_size, embedding_dim, padding_idx=padding_idx
        )
        self.convs = ModuleList([
            Conv2d(1, num_filters, (k, embedding_dim)) for k in kernel_sizes
        ])
        self.dropout = dropout
        self.dropout_layer = Dropout(dropout)
        self.fc = Linear(num_filters * len(kernel_sizes), output_dim)
        self.optimizer = AdamW(self.parameters(), lr=lr, weight_decay=1e-5)
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.output_dim = output_dim
        self.lr = lr

    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:
        x = self.embedding(x).unsqueeze(1)
        conv_x = [torch.relu(conv(x)).squeeze(3) for conv in self.convs]
        pooled_x = [torch.max(c, dim=2)[0] for c in conv_x]
        cat_x = torch.cat(pooled_x, dim=1)
        out = self.dropout_layer(cat_x)
        return self.fc(out)

    def train_model(
            self,
            train_X: torch.Tensor,
            train_y: torch.Tensor,
            val_X: torch.Tensor,
            val_y: torch.Tensor,
            num_epochs: int,
            batch_size: int,
    ) -> None:
        train_dataset = TensorDataset(train_X, train_y)
        val_dataset = TensorDataset(val_X, val_y)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        criterion = CrossEntropyLoss()

        for epoch in range(num_epochs):
            self.train()
            total_loss = 0.0
            num_batches = 0
            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()
                outputs = self(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / num_batches
            print(f"Epoch [{epoch + 1}/{num_epochs}]")
            print(f"Training avg Loss: {avg_loss:.4f}")

            self.eval()
            with torch.no_grad():
                total_val_loss = 0.0
                num_val_batches = 0
                for val_X_batch, val_y_batch in val_loader:
                    val_outputs = self(val_X_batch)
                    val_loss = criterion(val_outputs, val_y_batch)
                    total_val_loss += val_loss.item()
                    num_val_batches += 1

                avg_val_loss = total_val_loss / num_val_batches
                print(f"Validation avg Loss: {avg_val_loss:.4f}")

    # def predict(self, x: torch.Tensor) -> torch.Tensor:
    #     self.eval()
    #     with torch.no_grad():
    #         outputs = self(x)
    #         _, predicted = torch.max(outputs, 1)
    #     return predicted
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            max_kernel_size = max([conv.kernel_size[0] for conv in self.convs])
            if x.size(1) < max_kernel_size:
                pad_len = max_kernel_size - x.size(1)
                x = torch.nn.functional.pad(x, (0, pad_len), value=0)  # Pad at the end

            outputs = self(x)
            _, predicted = torch.max(outputs, 1)
        return predicted