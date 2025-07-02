from typing import Dict

import pandas as pd

from src.utils.enums import TopicClusteringConfig, ChatBotConfig
from src.utils.helpers import clean_text

topic_config = TopicClusteringConfig()
chatbot_config = ChatBotConfig()


def concat_documents(document_paths: Dict[str, str]) -> pd.DataFrame:
    dataframes = []
    for path, topic_label in document_paths.items():
        df = pd.read_csv(path)
        df[topic_config.LABEL_COL] = topic_label
        dataframes.append(df)

    return pd.concat(dataframes, ignore_index=True)


def encode_labels(df: pd.DataFrame) -> pd.DataFrame:
    label_mapping = chatbot_config.DOMAIN_ENCODE_MAPPING
    df[topic_config.LABEL_COL] = df[topic_config.LABEL_COL].map(label_mapping)
    df[topic_config.LABEL_COL] = df[topic_config.LABEL_COL].astype(int)
    return df


def ingest(
        document_paths: Dict[str, str],
        final_csv_dir: str,
) -> None:
    df = concat_documents(document_paths)
    df = df.dropna()
    df = df.reset_index(drop=True)
    df[topic_config.TEXT_COL] = df[topic_config.TEXT_COL].apply(clean_text)
    df[topic_config.TEXT_COL] = df[topic_config.TEXT_COL].str.lower()
    df = encode_labels(df)

    total_classes = df[topic_config.LABEL_COL].nunique()
    print(f"Total classes: {total_classes}")

    max_length = df[topic_config.TEXT_COL].apply(lambda x: len(x)).max()
    print(f"Max length: {max_length}")

    train_df = df[df["split"] == "train"]
    test_df = df[df["split"] == "test"]

    validation_df = test_df.iloc[:int(len(test_df) * 0.5)]
    new_test_df = test_df.iloc[int(len(test_df) * 0.5):]

    train_df.to_csv(f"{final_csv_dir}/training.csv", index=False)
    new_test_df.to_csv(f"{final_csv_dir}/test.csv", index=False)
    validation_df.to_csv(f"{final_csv_dir}/validation.csv", index=False)


if __name__ == "__main__":
    parent_path = "src/data/medical_data"

    all_paths = {
        f"{parent_path}/cancer/CancerQA.csv": chatbot_config.CANCER,
        f"{parent_path}/diabetes/Diabetes_and_Digestive_and_Kidney_DiseasesQA.csv": chatbot_config.DIABETES,
        f"{parent_path}/disease_control_and_prevention/Disease_Control_and_PreventionQA.csv": chatbot_config.DISEASE_CONTROL_AND_PREVENTION,
        f"{parent_path}/genetic_and_rare_diseases/Genetic_and_Rare_DiseasesQA.csv": chatbot_config.GENETIC_AND_RARE_DISEASES,
        f"{parent_path}/growth_hormone_receptor/growth_hormone_receptorQA.csv": chatbot_config.GROWTH_HORMONE_RECEPTOR,
        f"{parent_path}/heart_lung_and_blood/Heart_Lung_and_BloodQA.csv": chatbot_config.HEART_LUNG_AND_BLOOD,
        f"{parent_path}/neurological_disorders_and_stroke/Neurological_Disorders_and_StrokeQA.csv": chatbot_config.NEUROLOGICAL_DISORDERS_AND_STROKE,
        f"{parent_path}/others/OtherQA.csv": chatbot_config.OTHERS,
        f"{parent_path}/senior_health/SeniorHealthQA.csv": chatbot_config.SENIOR_HEALTH,
    }
    final_csv_dir = "src/data/medical_data/all"
    ingest(
        document_paths=all_paths,
        final_csv_dir=final_csv_dir,
    )
