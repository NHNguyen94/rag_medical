from typing import List

import pandas as pd
from llama_index.core import Document
from llama_index.readers.file import PandasCSVReader


class DocumentManager:
    def __init__(self):
        # self.csv_reader = PandasCSVReader()
        pass

    def load_csv_to_documents(self, csv_path: str, column_name: str) -> List[Document]:
        print(f"csv_path: {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"Length of csv: {len(df)}")
        print(f"5 first rows of csv: {df.head()}")
        print(f"5 last rows of csv: {df.tail()}")
        selected_data = df[column_name]
        # Ensure each row = 1 document
        documents = [Document(text=content) for content in selected_data]

        return documents
