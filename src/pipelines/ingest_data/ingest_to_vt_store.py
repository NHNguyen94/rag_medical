from src.services.ingestion_service import IngestionService
from src.utils.enums import IngestionConfig, ChatBotConfig


def ingest_single_index(
    single_data_path: str, single_index_path: str, col_name_to_ingest: str
) -> None:
    ingestion_service = IngestionService()
    ingestion_service.ingest_data(single_data_path, single_index_path, col_name_to_ingest)


def main(data_path: str, index_path: str, col_name_to_ingest) -> None:
    for _, v in ChatBotConfig.DOMAIN_MAPPING.items():
        single_index_path = f"{index_path}/{v}"
        single_data_path = f"{data_path}/{v}"
        ingest_single_index(single_data_path, single_index_path, col_name_to_ingest)


if __name__ == "__main__":
    index_path = IngestionConfig.INDEX_PATH
    data_path = IngestionConfig.DATA_PATH
    col_name_to_ingest = IngestionConfig.COL_NAME_TO_INGEST
    main(data_path, index_path, col_name_to_ingest)
