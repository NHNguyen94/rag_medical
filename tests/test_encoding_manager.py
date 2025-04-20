import torch

from src.core_managers.encoding_manager import EncodingManager


class TestEncodingManager:
    encoding_manager = EncodingManager()

    def test_tokenize_texts(self):
        texts = ["hello", "world"]
        tokenized_texts, max_length = self.encoding_manager.tokenize_texts(texts)
        assert tokenized_texts == [[104, 101, 108, 108, 111], [119, 111, 114, 108, 100]]
        assert max_length == 5

    def test_pad_sequences(self):
        tokenized_texts = [[104, 101, 108, 108, 111], [119, 111, 114, 108, 100]]
        padded_tokenized_texts = self.encoding_manager.pad_sequences(tokenized_texts)
        assert padded_tokenized_texts == [
            [104, 101, 108, 108, 111],
            [119, 111, 114, 108, 100],
        ]

    def test_to_tensor_float32(self):
        tokens = [1.0, 2.0, 3.0]
        tensor = self.encoding_manager.to_tensor(tokens, "float32")
        assert tensor.dtype == torch.float32
        assert tensor.tolist() == [1.0, 2.0, 3.0]
