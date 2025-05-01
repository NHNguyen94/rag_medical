from src.core_managers.document_manager import DocumentManager
from src.core_managers.vector_store_manager import VectorStoreManager
from src.utils.directory_manager import DirectoryManager
from src.utils.enums import IngestionConfig


class IngestionService:
    def __init__(self):
        self.doc_manager = DocumentManager()
        self.vt_store_manager = VectorStoreManager()

    def ingest_data(
        self, data_path: str, index_path: str, col_name_to_ingest: str
    ) -> None:
        all_csv_files = DirectoryManager.get_all_recursive_files(
            data_path, IngestionConfig.CSV_FILE_EXTENSION
        )
        print(f"All CSV files: {all_csv_files}")
        for csv_file in all_csv_files:
            print(f"Ingesting {csv_file}")
            documents = self.doc_manager.load_csv_to_documents(
                csv_file, col_name_to_ingest
            )
            if DirectoryManager.check_if_dir_exists(index_path):
                print(f"Loading index for {csv_file}")
                for doc in documents:
                    self.vt_store_manager.append_document(index_path, doc)
            else:
                print(f"Building index for {csv_file}")
                self.vt_store_manager.build_or_load_index(index_path, documents)
