from src.services.emotion_recognition_service import EmotionRecognitionService
from src.utils.directory_manager import DirectoryManager


def main(train_data_path: str, model_path: str):
    num_epochs = 20
    emotion_recognition_service = EmotionRecognitionService(
        use_embedding=True,
        embedding_dim=100,
        hidden_dim=128,
        layer_dim=3,
        lr=0.001,
        dropout=0.1,
    )
    if DirectoryManager.check_if_file_exists(model_path):
        DirectoryManager.delete_file(model_path)

    emotion_recognition_service.train_model(
        train_data_path=train_data_path,
        num_epochs=num_epochs,
        model_path=model_path,
        batch_size=16,
    )


if __name__ == "__main__":
    main(
        train_data_path="src/data/emotion_data/training.csv",
        model_path="src/ml_models/model_files/lstm_model.pth",
    )
