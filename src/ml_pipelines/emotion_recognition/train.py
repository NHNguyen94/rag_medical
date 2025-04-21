from src.services.emotion_recognition_service import EmotionRecognitionService

if __name__ == "__main__":
    num_epochs = 3
    emotion_recognition_service = EmotionRecognitionService(
        use_embedding=True,
        embedding_dim=300,
        hidden_dim=128,
        layer_dim=3,
        lr=0.001,
        dropout=0.2,
    )
    emotion_recognition_service.train_model(
        "src/data/emotion_data/train.csv",
        num_epochs
    )