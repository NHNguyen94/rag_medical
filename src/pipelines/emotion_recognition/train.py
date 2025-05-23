from src.services.emotion_recognition_service import EmotionRecognitionService
from src.utils.directory_manager import DirectoryManager


def run_train(train_data_path: str, model_path: str):
    num_epochs = 2
    emotion_recognition_service = EmotionRecognitionService(
        use_embedding=True,
        embedding_dim=128,
        hidden_dim=256,
        layer_dim=3,
        lr=0.0001,
        dropout=0.2,
    )
    if DirectoryManager.check_if_file_exists(model_path):
        DirectoryManager.delete_file(model_path)

    emotion_recognition_service.train_model(
        train_data_path=train_data_path,
        num_epochs=num_epochs,
        model_path=model_path,
        batch_size=16,
    )

def run_eval(model_path: str):
    emotion_recognition_service = EmotionRecognitionService(use_embedding=True)
    emotion_recognition_service.load_model(model_path)
    predictions = emotion_recognition_service.predict(["I am happy", "I am sad"])
    print(f"Predictions: {predictions}")
    emotion_recognition_service.evaluate_model("src/data/emotion_data/test.csv")

def main(train_data_path: str, model_path: str):
    run_train(
        train_data_path=train_data_path,
        model_path=model_path,
    )
    run_eval (
        model_path=model_path,
    )


if __name__ == "__main__":
    main(
        train_data_path="src/data/emotion_data/training.csv",
        model_path="src/ml_models/model_files/lstm_model.pth",
    )
