import argparse
from pathlib import Path
from src.pipelines.question_recommendation.fine_tune import FineTuningPipeline
from src.utils.enums import QuestionRecommendConfig


def main():
    # Configuration
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=int, required=True,
                        help='Dataset number (0=cancer, 1=diabetes, etc.)')
    args = parser.parse_args()
    model_name = "google/flan-t5-base"
    data_dir = QuestionRecommendConfig.DATASET_NUMBER_MAPPING[args.dataset]
    output_dir = QuestionRecommendConfig.MODEL_DATA_DIR
    save_dir = QuestionRecommendConfig.MODEL_PATH
    model_save_name = QuestionRecommendConfig.MODEL_SAVE_NAME[args.dataset]

    print(QuestionRecommendConfig.DATASET_NUMBER_MAPPING[args.dataset])
    print("dataset number:", args.dataset)
    print("model save name:" , model_save_name)


    # Initialize pipeline
    pipeline = FineTuningPipeline(
        model_name=model_name, data_dir=data_dir, output_dir=output_dir, model_save_path = save_dir, model_save_name= model_save_name
    )

    # Train model
    pipeline.train()


if __name__ == "__main__":
    main()