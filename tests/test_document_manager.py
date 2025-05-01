from llama_index.core import Document

from src.core_managers.document_manager import DocumentManager


class TestDocumentManager:
    doc_manager: DocumentManager

    def test_load_csv_to_documents(self):
        self.doc_manager = DocumentManager()
        csv_path = "tests/resources/test_data_for_loader/test.csv"
        documents = self.doc_manager.load_csv_to_documents(csv_path, "col1")
        assert len(documents) > 0
        assert isinstance(documents[0], Document)
