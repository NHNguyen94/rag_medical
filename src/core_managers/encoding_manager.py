from typing import List, Dict

import torch
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

from src.utils.enums import EmotionRecognitionConfig


class EncodingManager:
    def __init__(self, model_name="bert-base-uncased"):
        self.config = EmotionRecognitionConfig()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def yield_tokens(self, texts: List[str]):
        for text in texts:
            yield self.tokenizer(text)

    def build_vocab(self) -> Dict[str, int]:
        return self.tokenizer.get_vocab()

    def tokenize_text(self, text: str) -> List[int]:
        encodings = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.config.MAX_SEQ_LENGTH,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        tokenized_text = encodings["input_ids"]
        return tokenized_text

    def tokenize_texts(self, texts: List[str]) -> List[List[int]]:
        encodings = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config.MAX_SEQ_LENGTH,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        tokenized_texts = encodings["input_ids"]
        # max_length = max(len(seq) for seq in tokenized_texts)
        return tokenized_texts

    def to_tensor(self, tokens: List, data_type: str) -> Tensor:
        if data_type == self.config.FLOAT32:
            return torch.tensor(tokens, dtype=torch.float32)
        elif data_type == self.config.LONG:
            return torch.tensor(tokens, dtype=torch.long)
        else:
            raise ValueError("Unsupported data type")
