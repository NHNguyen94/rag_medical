import torch
from torch.nn import Module, LSTM, Linear, Embedding
import pandas as pd


class LSTMModel(Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            layer_dim: int,
            output_dim: int,
            vocab_size: int = 10000,
            embedding_dim: int = 100,
    ):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.embedding = Embedding(vocab_size, embedding_dim, padding_idx=0)
        print(f"self.embedding: {self.embedding}")
        self.lstm = LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.output_layer = Linear(hidden_dim, output_dim)

    def forward(
            self,
            x: torch.Tensor,
            h0: int = None,
            c0: int = None
    ):
        if h0 is None or c0 is None:
            h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
            c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)

        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.output_layer(out[:, -1, :])
        return out, hn, cn

num_classes = 6
model = LSTMModel(input_dim=1, hidden_dim=100, layer_dim=1, output_dim=num_classes)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
df = pd.read_csv("src/data/emotion_data/test.csv")
texts = df['text'].tolist()
labels = df["label"].tolist()
def tokenize_texts(texts):
    tokenized_texts = []
    for text in texts:
        tokenized_text = [ord(char) for char in text]
        tokenized_texts.append(tokenized_text)
    return tokenized_texts

tokenized_texts = tokenize_texts(texts)
max_length = max(len(text) for text in tokenized_texts)
for i in range(len(tokenized_texts)):
    tokenized_texts[i] += [0] * (max_length - len(tokenized_texts[i]))
trainX = torch.tensor(tokenized_texts, dtype=torch.float32)

unique_labels = sorted(set(labels))
label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
indexed_labels = [label_to_index[label] for label in labels]

trainY = torch.tensor(indexed_labels, dtype=torch.long)

trainX = trainX.view(-1, max_length, 1)

num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    outputs, _, _ = model(trainX)
    loss = criterion(outputs, trainY)

    loss.backward()
    optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Inference & Evaluation
model.eval()
with torch.no_grad():
    outputs, _, _ = model(trainX)
    predicted_classes = torch.argmax(outputs, dim=1)

    correct = (predicted_classes == trainY).sum().item()
    accuracy = correct / len(trainY)

    print(f"\nPredicted: {predicted_classes}")
    print(f"Actual:    {trainY}")
    print(f"Accuracy:  {accuracy:.4f}")

unique_predicted_labels = sorted(set(predicted_classes.tolist()))
print(f"Unique Predicted Labels: {unique_predicted_labels}")

# predicted_classes = torch.argmax(outputs, dim=1)
# print(f"Predicted: {predicted_classes}")
# print(f"Actual:    {trainY}")