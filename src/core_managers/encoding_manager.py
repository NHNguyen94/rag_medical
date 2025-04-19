from typing import List

import torch
from torch import Tensor

from src.utils.enums import LSTMConfig


class EncodingManager:
    def __init__(self):
        self.lstm_config = LSTMConfig()

    def tokenize_texts(self, texts: List[str]) -> (List[List[int]], int):
        tokenized_texts = []
        for text in texts:
            tokenized_text = [ord(char) for char in text]
            tokenized_texts.append(tokenized_text)
        max_length = max(len(text) for text in tokenized_texts)
        return tokenized_texts, max_length

    def pad_sequences(self, tokenized_texts: List[List[int]]) -> List[List[int]]:
        padded_tokenized_texts = []
        max_length = max(len(text) for text in tokenized_texts)
        for i in range(len(tokenized_texts)):
            padded_tokenized_texts.append(
                tokenized_texts[i] + [0] * (max_length - len(tokenized_texts[i]))
            )
        return padded_tokenized_texts

    def to_tensor(self, tokens: List, data_type: str) -> Tensor:
        if data_type == self.lstm_config.FLOAT32:
            return torch.tensor(tokens, dtype=torch.float32)
        elif data_type == self.lstm_config.LONG:
            return torch.tensor(tokens, dtype=torch.long)
        else:
            raise ValueError("Unsupported data type")
