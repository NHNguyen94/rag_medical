from typing import List

import pandas as pd
from llama_index.core import Document
from src.utils.helpers import clean_document_text


class DocumentManager:
    def __init__(self):
        # self.csv_reader = PandasCSVReader()
        pass

    def load_csv_to_documents(self, csv_path: str, column_name: str) -> List[Document]:
        print(f"csv_path: {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"Length of csv: {len(df)}")
        print(f"2 first rows of csv: {df.head(2)}")
        print(f"2 last rows of csv: {df.tail(2)}")
        selected_data = df[column_name]
        # Ensure each row = 1 document
        # documents = [Document(text=clean_document_text(content)) for content in selected_data]
        documents = []
        for content in selected_data:
            cleaned_content = clean_document_text(content)
            if cleaned_content:
                documents.append(Document(text=cleaned_content))
        print(f"\n\n\n2 first cleaned documents: {[doc.text for doc in documents[:5]]}\n\n\n")
        print(f"\n\n\n2 last cleaned documents: {[doc.text for doc in documents[-5:]]}\n\n\n")

        return documents
