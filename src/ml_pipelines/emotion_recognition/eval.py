from src.services.emotion_recognition_service import EmotionRecognitionService

if __name__ == "__main__":
    emotion_recognition_service = EmotionRecognitionService(use_embedding=True)
    emotion_recognition_service.load_model()
    predictions = emotion_recognition_service.predict(["I am happy", "I am sad"])
    print(f"Predictions: {predictions}")
    emotion_recognition_service.evaluate_model("src/data/emotion_data/test.csv")