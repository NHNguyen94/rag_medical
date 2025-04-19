from typing import List

import torch
from torch import Tensor


class EncodingManager():
    def __init__(self):
        pass

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
        if data_type == "float32":
            return torch.tensor(tokens, dtype=torch.float32)
        elif data_type == "long":
            return torch.tensor(tokens, dtype=torch.long)
        else:
            raise ValueError("Unsupported data type. Only 'float32' and 'long' are allowed.")
