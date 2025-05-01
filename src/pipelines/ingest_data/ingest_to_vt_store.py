from src.services.ingestion_service import IngestionService


def main(data_path: str, index_path: str) -> None:
    ingestion_service = IngestionService()
    ingestion_service.ingest_data(data_path, index_path)


if __name__ == "__main__":
    index_path = "src/indices"
    data_path = "src/data/medical_data"
    main(data_path, index_path)
