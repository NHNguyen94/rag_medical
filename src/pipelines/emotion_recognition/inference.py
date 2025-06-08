from src.services.emotion_recognition_service import EmotionRecognitionService
from src.utils.enums import ChatBotConfig


def run_inference(
    text: str,
) -> None:
    emotion_recognition_service = EmotionRecognitionService()
    model, vocab = emotion_recognition_service.load_model()
    predicted_emotion = emotion_recognition_service.predict(
        text=text,
        model=model,
        vocab=vocab,
    )
    predicted_emotion = int(predicted_emotion)
    predicted_emotion = ChatBotConfig.EMOTION_MAPPING[predicted_emotion]
    print(f"Input Text: {text}\nPredicted Emotion: {predicted_emotion}\n")


if __name__ == "__main__":
    input_texts = [
        "I am so happy today!",
        "I feel really sad and depressed.",
        "I'm a little bit worried about the future.",
        "I am angry about the situation.",
        "I am excited about the new project.",
    ]
    for text in input_texts:
        run_inference(text)
