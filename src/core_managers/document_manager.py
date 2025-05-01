from typing import List

import pandas as pd
from llama_index.core import Document
from llama_index.readers.file import PandasCSVReader


class DocumentManager:
    def __init__(self):
        self.csv_reader = PandasCSVReader()

    def load_csv_to_documents(self, csv_path: str, column_name: str) -> List[Document]:
        df = pd.read_csv(csv_path)
        selected_column = df[column_name]
        # Ensure each row = 1 document
        documents = [Document(text=content) for content in selected_column]

        return documents
