import re
from typing import List, Dict
from collections import defaultdict
import torch
from torch.nn.utils.rnn import pad_sequence
from torch import Tensor
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from src.utils.enums import LSTMConfig


class EncodingManager:
    def __init__(self):
        self.lstm_config = LSTMConfig()
        self.tokenizer = get_tokenizer("basic_english")

    def yield_tokens(self, texts: List[str]):
        for text in texts:
            yield self.tokenizer(text)

    def build_vocab(self, texts: List[str], min_freq: int = 1) -> Dict[str, int]:
        vocab = build_vocab_from_iterator(
            self.yield_tokens(texts), min_freq=min_freq, specials=["<pad>"]
        )
        vocab.set_default_index(vocab["<unk>"])
        return vocab



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

    def to_tensor(
            self,
            tokens: List,
            data_type: str
    ) -> Tensor:
        if data_type == self.lstm_config.FLOAT32:
            return torch.tensor(tokens, dtype=torch.float32)
        elif data_type == self.lstm_config.LONG:
            return torch.tensor(tokens, dtype=torch.long)
        else:
            raise ValueError("Unsupported data type")


if __name__ == "__main__":
    # Example usage
    encoding_manager = EncodingManager()
    texts = ["hello", "world"]
    # build vocab
    vocab = encoding_manager.build_vocab(texts)
    print("Vocab:", vocab)