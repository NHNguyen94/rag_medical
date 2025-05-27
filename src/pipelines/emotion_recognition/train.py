from datetime import datetime
from typing import Dict

from src.services.emotion_recognition_service import EmotionRecognitionService
from src.utils.directory_manager import DirectoryManager
from src.utils.helpers import write_log_file


def run_train(
        train_data_path: str,
        validation_data_path: str,
        model_path: str,
        num_epochs: int,
        use_embedding: bool,
        embedding_dim: int,
        hidden_dim: int,
        layer_dim: int,
        lr: float,
        dropout: float,
        batch_size: int,
) -> None:
    emotion_recognition_service = EmotionRecognitionService(
        use_embedding=use_embedding,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        layer_dim=layer_dim,
        lr=lr,
        dropout=dropout,
    )

    emotion_recognition_service.train_model(
        train_data_path=train_data_path,
        validation_data_path=validation_data_path,
        num_epochs=num_epochs,
        model_path=model_path,
        batch_size=batch_size,
    )


def run_eval(model_path: str, eval_data_path: str) -> Dict:
    emotion_recognition_service = EmotionRecognitionService(use_embedding=True)
    emotion_recognition_service.load_model(model_path)

    return emotion_recognition_service.evaluate_model(eval_data_path)


def main(
        train_data_path: str,
        eval_data_path: str,
        model_path: str,
        log_path: str,
        log_file_name: str,
        num_epochs: int,
        use_embedding: bool,
        embedding_dim: int,
        hidden_dim: int,
        layer_dim: int,
        lr: float,
        dropout: float,
        batch_size: int,
) -> None:
    start_time = datetime.now()
    run_train(
        train_data_path=train_data_path,
        validation_data_path=eval_data_path,
        model_path=model_path,
        num_epochs=num_epochs,
        use_embedding=use_embedding,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        layer_dim=layer_dim,
        lr=lr,
        dropout=dropout,
        batch_size=batch_size,
    )
    end_time = datetime.now()

    eval_result = run_eval(
        model_path=model_path,
        eval_data_path=eval_data_path,
    )

    eval_result["start_time"] = start_time
    eval_result["end_time"] = end_time
    eval_result["training_time"] = (end_time - start_time).total_seconds()
    eval_result["num_epochs"] = num_epochs
    eval_result["use_embedding"] = use_embedding
    eval_result["embedding_dim"] = embedding_dim
    eval_result["hidden_dim"] = hidden_dim
    eval_result["layer_dim"] = layer_dim
    eval_result["lr"] = lr
    eval_result["dropout"] = dropout
    eval_result["batch_size"] = batch_size
    eval_result["train_data_path"] = train_data_path
    eval_result["eval_data_path"] = eval_data_path
    eval_result["model_path"] = model_path
    print(f"Evaluation Result: {eval_result}")

    full_log_path = f"{log_path}/{log_file_name}"
    if not DirectoryManager.check_if_file_exists(full_log_path):
        col_names = [col_name for col_name in eval_result.keys()]
        DirectoryManager.create_empty_csv_file(
            col_names=col_names, file_path=full_log_path
        )

    write_log_file(full_log_path, eval_result, False)


if __name__ == "__main__":
    # train_data_path = "src/data/emotion_data/training.csv"
    train_data_path = "src/data/emotion_data/training_small.csv"
    eval_data_path = "src/data/emotion_data/training_small.csv"
    model_path = "src/ml_models/model_files/lstm_model.pth"
    log_path = "src/data/training_logs"
    log_file_name = "emotion_recognition.csv"
    DirectoryManager.create_dir_if_not_exists(log_path)
    if DirectoryManager.check_if_file_exists(model_path):
        DirectoryManager.delete_file(model_path)

    main(
        train_data_path=train_data_path,
        eval_data_path=eval_data_path,
        model_path=model_path,
        log_path=log_path,
        log_file_name=log_file_name,
        num_epochs=30,
        use_embedding=True,
        embedding_dim=128,
        hidden_dim=256,
        layer_dim=3,
        lr=0.0005,
        dropout=0.1,
        batch_size=8,
    )
