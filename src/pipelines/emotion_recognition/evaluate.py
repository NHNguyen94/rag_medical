from typing import Dict

from src.services.emotion_recognition_service import EmotionRecognitionService


def run_eval(model_path: str, test_data_path: str) -> Dict:
    emotion_recognition_service = EmotionRecognitionService(
        test_data_path=test_data_path,
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
    model_path: str,
    test_data_path: str,
) -> None:
    eval_result = run_eval(model_path, test_data_path)
    print(f"Evaluation Result: {eval_result}")


if __name__ == "__main__":
    model_path = "src/ml_models/model_files/cnn_model.pth"
    test_data_path = "src/data/emotion_data/test.csv"

    main(model_path, test_data_path)
