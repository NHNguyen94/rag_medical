from audioop import avgpp

import torch
from torch.nn import Module, LSTM, Linear, Embedding, CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset


class LSTMModel(Module):
    def __init__(
        self,
        hidden_dim: int,
        layer_dim: int,
        output_dim: int,
        input_dim: int = None,
        vocab_size: int = 10000,
        embedding_dim: int = 100,
        lr: float = 0.01,
        dropout: float = 0.2,
    ):
        super(LSTMModel, self).__init__()
        # Select between input_dim and embedding_dim
        self.input_dim = input_dim
        if self.input_dim:
            # print("Using input_dim")
            self.lstm = LSTM(
                input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout
            )
        else:
            # print("Using embedding_dim")
            self.embedding = Embedding(vocab_size, embedding_dim, padding_idx=0)
            self.lstm = LSTM(embedding_dim, hidden_dim, layer_dim, batch_first=True)
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        # self.embedding = Embedding(vocab_size, embedding_dim, padding_idx=0)
        # self.lstm = LSTM(embedding_dim, hidden_dim, layer_dim, batch_first=True)
        self.output_layer = Linear(hidden_dim, output_dim)
        self.criterion = CrossEntropyLoss()
        self.optimizer = Adam(self.parameters(), lr=lr, weight_decay=1e-5)

    def create_dataloader(
        self, X_train: torch.Tensor, y_train: torch.Tensor, batch_size: int
    ) -> DataLoader:
        dataset = TensorDataset(X_train, y_train)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return dataloader

    def forward(self, x: torch.Tensor, h0: int = None, c0: int = None):
        if self.input_dim:
            # print(f"x.dim(): {x.dim()}")
            # print(f"x.size(): {x.size()}")
            if h0 is None or c0 is None:
                h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(
                    x.device
                )
                c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(
                    x.device
                )

            out, (hn, cn) = self.lstm(x, (h0, c0))
            out = self.output_layer(out[:, -1, :])
            return out, hn, cn
        else:
            # print(f"x.dim(): {x.dim()}")
            # print(f"x.size(): {x.size()}")
            x = self.embedding(x)
            # x = x.squeeze(2)
            if h0 is None or c0 is None:
                h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(
                    x.device
                )
                c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(
                    x.device
                )
            out, (hn, cn) = self.lstm(x, (h0, c0))
            out = self.output_layer(out[:, -1, :])
            return out, hn, cn

    def train_model(
            self,
            trainX: torch.Tensor,
            trainY: torch.Tensor,
            num_epochs: int,
            batch_size: int,
    ) -> None:
        dataloader = self.create_dataloader(trainX, trainY, batch_size)

        for epoch in range(num_epochs):
            self.train()
            total_loss = 0.0
            num_batches = 0
            for batch_X, batch_Y in dataloader:
                self.optimizer.zero_grad()
                outputs, _, _ = self(batch_X)
                loss = self.criterion(outputs, batch_Y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                num_batches += 1
            avg_loss = total_loss / num_batches
            print(f"Epoch [{epoch + 1}/{num_epochs}], Avg Loss: {avg_loss:.4f}")

    def predict_class(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            outputs, _, _ = self(x)
            predicted_classes = torch.argmax(outputs, dim=1)
        return predicted_classes

    def evaluate_model(self, x: torch.Tensor, y: torch.Tensor) -> None:
        predicted_classes = self.predict_class(x)

        correct = (predicted_classes == y).sum().item()
        accuracy = correct / len(y)

        print(f"\nPredicted: {predicted_classes}")
        print(f"Actual:    {y}")
        print(f"Accuracy:  {accuracy:.4f}")

        unique_predicted_labels = sorted(set(predicted_classes.tolist()))
        print(f"Unique Predicted Labels: {unique_predicted_labels}")
