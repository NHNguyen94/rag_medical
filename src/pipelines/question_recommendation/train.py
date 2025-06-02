from pathlib import Path
from src.pipelines.question_recommendation.fine_tune import FineTuningPipeline


def main():
    # Configuration
    model_name = "google/flan-t5-base"
    data_dir = "../../data/fine_tune_dataset"
    output_dir = "./../ml_modles/flant5"

    # Initialize pipeline
    pipeline = FineTuningPipeline(
        model_name=model_name,
        data_dir=data_dir,
        output_dir=output_dir
    )

    # Train model
    pipeline.train()


if __name__ == "__main__":
    main()