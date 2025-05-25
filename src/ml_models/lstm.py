import torch
from torch.nn import (
    Module,
    Linear,
    Embedding,
    CrossEntropyLoss,
    ModuleList,
    Dropout,
    # LSTMCell
)
from torch.optim import AdamW, Adam
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils import clip_grad_norm_
from torch.nn.init import xavier_uniform_, zeros_

TRAINING_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LSTMCell(Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super(LSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.W_i = Linear(input_dim, hidden_dim)
        self.U_i = Linear(hidden_dim, hidden_dim)

        self.W_f = Linear(input_dim, hidden_dim)
        self.U_f = Linear(hidden_dim, hidden_dim)

        self.W_o = Linear(input_dim, hidden_dim)
        self.U_o = Linear(hidden_dim, hidden_dim)

        self.W_c = Linear(input_dim, hidden_dim)
        self.U_c = Linear(hidden_dim, hidden_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        for layer in [
            self.W_i,
            self.U_i,
            self.W_f,
            self.U_f,
            self.W_o,
            self.U_o,
            self.W_c,
            self.U_c,
        ]:
            xavier_uniform_(layer.weight)
            if layer.bias is not None:
                zeros_(layer.bias)

    def forward(
        self, x_t: torch.Tensor, h_prev: torch.Tensor, c_prev: torch.Tensor
    ) -> (torch.Tensor, torch.Tensor):
        i_t = torch.sigmoid(self.W_i(x_t) + self.U_i(h_prev))
        f_t = torch.sigmoid(self.W_f(x_t) + self.U_f(h_prev))
        o_t = torch.sigmoid(self.W_o(x_t) + self.U_o(h_prev))
        c_hat = torch.tanh(self.W_c(x_t) + self.U_c(h_prev))

        c_t = f_t * c_prev + i_t * c_hat
        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t


class CustomLSTMCell(Module):
    def __init__(self, input_dim: int, hidden_dim: int, layer_dim: int, dropout: float):
        super(CustomLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.dropout = dropout

        self.lstm_cells = ModuleList(
            [
                LSTMCell(input_dim if i == 0 else hidden_dim, hidden_dim)
                for i in range(layer_dim)
            ]
        )
        self.dropout_layer = Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        hx: (torch.Tensor, torch.Tensor),
    ) -> (torch.Tensor, (list, list)):
        h_n, c_n = [], []
        h_prev, c_prev = hx

        for i, cell in enumerate(self.lstm_cells):
            h_i, c_i = cell(x, h_prev[i], c_prev[i])
            x = self.dropout_layer(h_i)
            # x = h_i
            h_n.append(h_i)
            c_n.append(c_i)

        return x, (h_n, c_n)


class CustomLSTMLayer(Module):
    def __init__(self, input_dim: int, hidden_dim: int, layer_dim: int, dropout: float):
        super(CustomLSTMLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.cell = CustomLSTMCell(input_dim, hidden_dim, layer_dim, dropout)

    def forward(
        self, x: torch.Tensor, hx: (torch.Tensor, torch.Tensor) = None
    ) -> (torch.Tensor, (list, list)):
        batch_size, seq_len, _ = x.size()

        if hx is None:
            h_0 = [
                torch.zeros(batch_size, self.cell.hidden_dim, device=x.device)
                for _ in range(self.cell.layer_dim)
            ]
            c_0 = [
                torch.zeros(batch_size, self.cell.hidden_dim, device=x.device)
                for _ in range(self.cell.layer_dim)
            ]
        else:
            h_0, c_0 = hx

        outputs = []
        h_t, c_t = h_0, c_0

        for t in range(seq_len):
            x_t = x[:, t, :]  # (batch_size, input_dim)
            out_t, (h_t, c_t) = self.cell(x_t, (h_t, c_t))
            outputs.append(out_t.unsqueeze(1))  # (batch_size, 1, hidden_dim)

        outputs = torch.cat(outputs, dim=1)  # (batch_size, seq_len, hidden_dim)
        return outputs, (h_t, c_t)


class LSTMModel(Module):
    def __init__(
        self,
        hidden_dim: int,
        layer_dim: int,
        output_dim: int,
        lr: float,
        input_dim: int = None,
        vocab_size: int = 10000,
        embedding_dim: int = 100,
        dropout: float = 0.2,
        padding_idx: int = 0,
    ):
        super(LSTMModel, self).__init__()
        # Select between input_dim and embedding_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.lr = lr
        if self.input_dim:
            # print("Using input_dim")
            # self.lstm = LSTM(
            #     input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout
            # )
            self.lstm = CustomLSTMLayer(input_dim, hidden_dim, layer_dim, dropout)
        else:
            # print("Using embedding_dim")
            self.embedding = Embedding(
                vocab_size, embedding_dim, padding_idx=padding_idx
            )
            # self.lstm = LSTM(embedding_dim, hidden_dim, layer_dim, batch_first=True)
            self.lstm = CustomLSTMLayer(embedding_dim, hidden_dim, layer_dim, dropout)
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.output_layer = Linear(hidden_dim, output_dim)
        # self.criterion = CrossEntropyLoss()
        self.optimizer = AdamW(self.parameters(), lr=lr, weight_decay=1e-5)
        # self.optimizer = Adam(self.parameters(), lr=lr, weight_decay=1e-5)

    def compute_class_weights(self, y_train: torch.Tensor) -> torch.Tensor:
        class_counts = torch.bincount(y_train, minlength=self.output_dim).float()
        print(f"Class counts: {class_counts}")
        weights = 1.0 / (class_counts + 1e-6)
        weights = weights * (self.output_dim / weights.sum())
        print(f"Weights after scaling: {weights}")
        return weights

    def get_criterion(self, y_train: torch.Tensor) -> CrossEntropyLoss:
        class_weights = self.compute_class_weights(y_train)
        return CrossEntropyLoss(weight=class_weights.to(TRAINING_DEVICE))

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
                    TRAINING_DEVICE
                )
                c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(
                    TRAINING_DEVICE
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
                    TRAINING_DEVICE
                )
                c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(
                    TRAINING_DEVICE
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
        trainX = trainX.to(TRAINING_DEVICE)
        trainY = trainY.to(TRAINING_DEVICE)

        dataloader = self.create_dataloader(trainX, trainY, batch_size)
        criterion = self.get_criterion(trainY)

        for epoch in range(num_epochs):
            self.train()
            total_loss = 0.0
            num_batches = 0
            for batch_X, batch_Y in dataloader:
                batch_X = batch_X.to(TRAINING_DEVICE)
                batch_Y = batch_Y.to(TRAINING_DEVICE)

                self.optimizer.zero_grad()
                outputs, _, _ = self(batch_X)
                loss = criterion(outputs, batch_Y)
                loss.backward()
                clip_grad_norm_(self.parameters(), max_norm=5)
                self.optimizer.step()
                total_loss += loss.item()
                num_batches += 1
                # print("Batch output mean/std:", outputs.mean().item(), outputs.std().item())
            avg_loss = total_loss / num_batches
            print(f"Epoch [{epoch + 1}/{num_epochs}], Avg Loss: {avg_loss:.4f}")

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(TRAINING_DEVICE)
        self.eval()
        with torch.no_grad():
            outputs, _, _ = self(x)
            predicted_classes = torch.argmax(outputs, dim=1)
        return predicted_classes
