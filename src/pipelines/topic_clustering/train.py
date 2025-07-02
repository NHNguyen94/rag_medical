from datetime import datetime
from typing import Dict

from src.services.topic_clustering_service import TopicClusteringService
from src.utils.directory_manager import DirectoryManager
from src.utils.enums import TopicClusteringConfig

topic_config = TopicClusteringConfig()


def run_train(
    model_name: str,
    batch_size: int,
    epochs: int,
    train_data_path: str,
    test_data_path: str,
    validation_data_path: str,
    model_path: str,
    lr: float,
) -> (float, float):
    topic_clustering_service = TopicClusteringService(
        model_name=model_name,
        train_data_path=train_data_path,
        test_data_path=test_data_path,
        validation_data_path=validation_data_path,
        lr=lr,
    )
    train_loss, val_loss = topic_clustering_service.train(
        batch_size=batch_size, epochs=epochs
    )
    topic_clustering_service.save_model(model_path)

    return train_loss, val_loss


def run_eval(validation_data_path: str, model_path: str) -> Dict:
    topic_clustering_service = TopicClusteringService(
        validation_data_path=validation_data_path
    )
    model = topic_clustering_service.load_model(model_path)
    eval_result = topic_clustering_service.evaluate(model)
    return eval_result


def main(
    model_name: str,
    batch_size: int,
    epochs: int,
    train_data_path: str,
    test_data_path: str,
    validation_data_path: str,
    model_path: str,
    lr: float,
    log_path: str,
    log_file_name: str,
) -> None:
    start_time = datetime.now()
    train_loss, val_loss = run_train(
        model_name=model_name,
        batch_size=batch_size,
        epochs=epochs,
        train_data_path=train_data_path,
        test_data_path=test_data_path,
        validation_data_path=validation_data_path,
        model_path=model_path,
        lr=lr,
    )
    end_time = datetime.now()
    eval_result = run_eval(
        validation_data_path=validation_data_path, model_path=model_path
    )

    eval_result["train_loss"] = train_loss
    eval_result["val_loss"] = val_loss
    eval_result["start_time"] = start_time
    eval_result["end_time"] = end_time
    eval_result["training_time"] = (end_time - start_time).total_seconds()
    eval_result["num_epochs"] = epochs
    eval_result["batch_size"] = batch_size
    eval_result["model_name"] = model_name
    eval_result["model_path"] = model_path
    eval_result["lr"] = lr

    full_log_path = f"{log_path}/{log_file_name}"
    DirectoryManager.write_log_file(full_log_path, eval_result)


if __name__ == "__main__":
    num_classes = 9
    train_data_path = "src/data/medical_data/all/training.csv"
    test_data_path = "src/data/medical_data/all/test.csv"
    validation_data_path = "src/data/medical_data/all/validation.csv"
    model_path = "src/ml_models/model_files/topic_clustering_model.pth"
    log_path = "src/data/training_logs"
    log_file_name = "topic_clustering.csv"

    DirectoryManager.create_dir_if_not_exists(log_path)
    if DirectoryManager.check_if_file_exists(model_path):
        DirectoryManager.delete_file(model_path)

    hyper_params = [
        {
            "model_name": "google/bert_uncased_L-2_H-128_A-2",
            "batch_size": 32,
            "epochs": 150,
            "lr": 0.000005,
        },
        # {
        #     "model_name": "google/bert_uncased_L-4_H-256_A-4",
        #     "batch_size": 32,
        #     "epochs": 10,
        #     "lr": 0.00001,
        # },
        # {
        #     "model_name": "google/bert_uncased_L-6_H-512_A-8",
        #     "batch_size": 32,
        #     "epochs": 5,
        #     "lr": 0.001,
        # },
        # {
        #     "model_name": "google/bert_uncased_L-8_H-512_A-8",
        #     "batch_size": 32,
        #     "epochs": 5,
        #     "lr": 0.001,
        # },
    ]

    for params in hyper_params:
        main(
            model_name=params["model_name"],
            batch_size=params["batch_size"],
            epochs=params["epochs"],
            train_data_path=train_data_path,
            test_data_path=test_data_path,
            validation_data_path=validation_data_path,
            model_path=model_path,
            lr=params["lr"],
            log_path=log_path,
            log_file_name=log_file_name,
        )
