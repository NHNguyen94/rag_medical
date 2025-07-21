from typing import Dict

from src.services.topic_clustering_service import TopicClusteringService


def run_eval(test_data_path: str, model_path: str) -> Dict:
    topic_clustering_service = TopicClusteringService(test_data_path=test_data_path)
    model = topic_clustering_service.load_model(model_path)
    eval_result = topic_clustering_service.evaluate(model)
    return eval_result


def main(test_data_path: str, model_path: str) -> None:
    eval_result = run_eval(test_data_path, model_path)
    print(f"Evaluation Result: {eval_result}")


if __name__ == "__main__":
    test_data_path = "src/data/medical_data/all/test.csv"
    model_path = "src/ml_models/model_files/topic_clustering_model.pth"

    main(test_data_path, model_path)
