from datetime import datetime
from typing import Dict, List

from src.services.emotion_recognition_service import EmotionRecognitionService
from src.utils.directory_manager import DirectoryManager


def run_train(
    train_data_path: str,
    test_data_path: str,
    validation_data_path: str,
    model_path: str,
    embedding_dim: int,
    num_classes: int,
    kernel_sizes: list,
    num_filters: int,
    dropout: float,
    lr: float,
    batch_size: int,
    epochs: int,
) -> (float, float):
    emotion_recognition_service = EmotionRecognitionService(
        train_data_path=train_data_path,
        test_data_path=test_data_path,
        validation_data_path=validation_data_path,
        embed_dim=embedding_dim,
        num_classes=num_classes,
        kernel_sizes=kernel_sizes,
        num_filters=num_filters,
        dropout=dropout,
        lr=lr,
    )

    (train_loss, val_loss) = emotion_recognition_service.train(
        batch_size=batch_size,
        epochs=epochs,
    )

    emotion_recognition_service.save_model(
        model_path=model_path,
    )

    return train_loss, val_loss


def run_eval(model_path: str, validation_data_path: str) -> Dict:
    emotion_recognition_service = EmotionRecognitionService(
        validation_data_path=validation_data_path,
    )
    model, vocab = emotion_recognition_service.load_model(
        model_path=model_path,
    )
    eval_result = emotion_recognition_service.evaluate(
        model=model,
        vocab=vocab,
    )
    return eval_result


def main(
    train_data_path: str,
    test_data_path: str,
    validation_data_path: str,
    model_path: str,
    log_path: str,
    log_file_name: str,
    embedding_dim: int,
    num_classes: int,
    kernel_sizes: List,
    num_filters: int,
    dropout: float,
    lr: float,
    batch_size: int,
    num_epochs: int,
) -> None:
    start_time = datetime.now()
    train_loss, val_loss = run_train(
        train_data_path=train_data_path,
        test_data_path=test_data_path,
        validation_data_path=validation_data_path,
        model_path=model_path,
        embedding_dim=embedding_dim,
        num_classes=num_classes,
        kernel_sizes=kernel_sizes,
        num_filters=num_filters,
        dropout=dropout,
        lr=lr,
        batch_size=batch_size,
        epochs=num_epochs,
    )
    end_time = datetime.now()

    eval_result = run_eval(
        model_path=model_path,
        validation_data_path=validation_data_path,
    )

    eval_result["train_loss"] = train_loss
    eval_result["val_loss"] = val_loss
    eval_result["start_time"] = start_time
    eval_result["end_time"] = end_time
    eval_result["training_time"] = (end_time - start_time).total_seconds()
    eval_result["num_epochs"] = num_epochs
    eval_result["embedding_dim"] = embedding_dim
    eval_result["num_classes"] = num_classes
    eval_result["kernel_sizes"] = kernel_sizes
    eval_result["num_filters"] = num_filters
    eval_result["lr"] = lr
    eval_result["dropout"] = dropout
    eval_result["batch_size"] = batch_size
    eval_result["train_data_path"] = train_data_path
    eval_result["test_data_path"] = test_data_path
    eval_result["validation_data_path"] = validation_data_path
    eval_result["model_path"] = model_path
    print(f"Evaluation Result: {eval_result}")

    full_log_path = f"{log_path}/{log_file_name}"
    DirectoryManager.write_log_file(full_log_path, eval_result)


if __name__ == "__main__":
    train_data_path = "src/data/emotion_data/training.csv"
    test_data_path = "src/data/emotion_data/test.csv"
    validation_data_path = "src/data/emotion_data/validation.csv"
    model_path = "src/ml_models/model_files/cnn_model.pth"
    log_path = "src/data/training_logs"
    log_file_name = "emotion_recognition.csv"
    num_classes = 6
    DirectoryManager.create_dir_if_not_exists(log_path)
    if DirectoryManager.check_if_file_exists(model_path):
        DirectoryManager.delete_file(model_path)

    hyper_params = [
        {
            "embedding_dim": 300,
            "kernel_sizes": [2, 3, 4, 5],
            "num_filters": 200,
            "dropout": 0.5,
            "lr": 0.001,
            "batch_size": 32,
            "num_epochs": 10,
        },
    ]

    for params in hyper_params:
        main(
            train_data_path=train_data_path,
            test_data_path=test_data_path,
            validation_data_path=validation_data_path,
            model_path=model_path,
            log_path=log_path,
            log_file_name=log_file_name,
            embedding_dim=params["embedding_dim"],
            num_classes=num_classes,
            kernel_sizes=params["kernel_sizes"],
            num_filters=params["num_filters"],
            dropout=params["dropout"],
            lr=params["lr"],
            batch_size=params["batch_size"],
            num_epochs=params["num_epochs"],
        )
