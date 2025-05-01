from src.services.ingestion_service import IngestionService
from src.utils.enums import IngestionConfig


def main(data_path: str, index_path: str, col_name_to_ingest: str) -> None:
    ingestion_service = IngestionService()
    ingestion_service.ingest_data(data_path, index_path, col_name_to_ingest)


if __name__ == "__main__":
    index_path = IngestionConfig.INDEX_PATH
    data_path = IngestionConfig.DATA_PATH
    col_name_to_ingest = IngestionConfig.COL_NAME_TO_INGEST
    main(data_path, index_path, col_name_to_ingest)
