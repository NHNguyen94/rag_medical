from pathlib import Path
from typing import List

from llama_index.core import Document
from llama_index.readers.file import PandasCSVReader


class DocumentManager:
    def __init__(self):
        self.csv_reader = PandasCSVReader()

    def load_csv_to_documents(self, csv_path: str) -> List[Document]:
        return self.csv_reader.load_data(Path(csv_path))
