import torch

from src.core_managers.encoding_manager import EncodingManager


class TestEncodingManager:
    encoding_manager = EncodingManager()

    def test_tokenize_texts(self):
        texts = ["hello", "world"]
        tokenized_texts, max_length = self.encoding_manager.tokenize_texts(texts)
        assert tokenized_texts == [[101, 7592, 102], [101, 2088, 102]]
        assert max_length == 3

    def test_to_tensor_float32(self):
        tokens = [1.0, 2.0, 3.0]
        tensor = self.encoding_manager.to_tensor(tokens, "float32")
        assert tensor.dtype == torch.float32
        assert tensor.tolist() == [1.0, 2.0, 3.0]

    def test_build_vocab(self):
        vocab = self.encoding_manager.build_vocab()
        print("Vocab sample:", dict(list(vocab.items())[:10]))
