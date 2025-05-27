import torch

from src.core_managers.encoding_manager import EncodingManager


class TestEncodingManager:
    encoding_manager = EncodingManager()

    def test_tokenize_text(self):
        text = "hello world"
        tokenized_text = self.encoding_manager.tokenize_text(text)
        # print("Tokenized texts:", tokenized_text)
        assert tokenized_text == [101, 7592, 2088, 102]

    def test_tokenize_texts(self):
        texts = ["hello", "world"]
        tokenized_texts = self.encoding_manager.tokenize_texts(texts)
        # print("Tokenized texts:", tokenized_texts)
        assert tokenized_texts == [[101, 7592, 102], [101, 2088, 102]]

    def test_to_tensor_float32(self):
        tokens = [1.0, 2.0, 3.0]
        tensor = self.encoding_manager.to_tensor(tokens, "float32")
        assert tensor.dtype == torch.float32
        assert tensor.tolist() == [1.0, 2.0, 3.0]

    def test_build_vocab(self):
        vocab = self.encoding_manager.build_vocab()
        print("Vocab sample:", dict(list(vocab.items())[:10]))
