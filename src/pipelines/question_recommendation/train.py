from pathlib import Path
from src.pipelines.question_recommendation.fine_tune import FineTuningPipeline
from src.utils.enums import QuestionRecommendConfig


def main():
    # Configuration
    model_name = "google/flan-t5-base"
    data_dir = QuestionRecommendConfig.FINE_TUNE_DATA_DIR
    output_dir = QuestionRecommendConfig.MODEL_DIR

    # Initialize pipeline
    pipeline = FineTuningPipeline(
        model_name=model_name, data_dir=data_dir, output_dir=output_dir
    )

    # Train model
    pipeline.train()


if __name__ == "__main__":
    main()
