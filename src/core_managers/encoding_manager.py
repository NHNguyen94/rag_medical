from typing import List, Dict

import torch
from torch import Tensor
from transformers import AutoTokenizer

from src.utils.enums import LSTMConfig


class EncodingManager:
    def __init__(self, model_name="bert-base-uncased"):
        self.lstm_config = LSTMConfig()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def yield_tokens(self, texts: List[str]):
        for text in texts:
            yield self.tokenizer(text)

    def build_vocab(self, texts: List[str]) -> Dict[str, int]:
        return self.tokenizer.get_vocab()

    def tokenize_texts(self, texts: List[str]) -> (List[List[int]], int):
        encodings = self.tokenizer(
            texts,
            padding=True,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        tokenized_texts = encodings["input_ids"]
        max_length = max(len(seq) for seq in tokenized_texts)
        return tokenized_texts, max_length

    def to_tensor(self, tokens: List, data_type: str) -> Tensor:
        if data_type == self.lstm_config.FLOAT32:
            return torch.tensor(tokens, dtype=torch.float32)
        elif data_type == self.lstm_config.LONG:
            return torch.tensor(tokens, dtype=torch.long)
        else:
            raise ValueError("Unsupported data type")
