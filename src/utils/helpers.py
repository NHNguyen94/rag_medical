import hashlib
import re
import textwrap
from collections import Counter
from typing import Dict, List
from uuid import UUID, uuid4

import nltk
import pandas as pd
import yaml
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.metrics import confusion_matrix

from src.utils.enums import ChatBotConfig

TRAIN_LOG_PATH = "src/data/training_logs"


def load_yml_configs(config_path: str) -> Dict:
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def write_dict_to_yaml(data: Dict, file_path: str) -> None:
    with open(file_path, "w") as file:
        yaml.dump(data, file, default_flow_style=False, sort_keys=False)


def get_unique_id() -> UUID:
    return uuid4()


def hash_string(string: str) -> str:
    return hashlib.sha256(string.encode()).hexdigest()


def clean_document_text(doc: str) -> str:
    cleaned = "\n".join(line.strip() for line in doc.strip().splitlines())
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return textwrap.dedent(cleaned).strip()


def sample_qa_data() -> List[Dict]:
    return [
        {
            "question": "What are the symptoms of diabetes?",
            "answer": "Common symptoms include increased thirst, frequent urination, fatigue, and blurred vision.",
            "topic": "diabetes",
        },
        {
            "question": "How is type 2 diabetes treated?",
            "answer": "Treatment includes lifestyle changes, oral medications like metformin, and sometimes insulin therapy.",
            "topic": "diabetes",
        },
        {
            "question": "What causes lung cancer?",
            "answer": "Lung cancer is primarily caused by smoking, but also by exposure to radon, asbestos, or air pollution.",
            "topic": "cancer",
        },
        {
            "question": "What are the signs of breast cancer?",
            "answer": "Signs include a lump in the breast, changes in breast shape, or nipple discharge.",
            "topic": "cancer",
        },
        {
            "question": "How can I prevent heart disease?",
            "answer": "Prevent heart disease by eating a healthy diet, exercising regularly, and avoiding smoking.",
            "topic": "heart disease",
        },
    ]


def download_nlkt() -> None:
    nltk.download("punkt")
    nltk.download("punkt_tab")
    nltk.download("stopwords")


def clean_text(text: str) -> str:
    text = clean_document_text(text)
    text = text.lower().strip()
    text = re.sub(r"\d+", "", text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalpha()]
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words]

    return " ".join(tokens)


def clean_and_tokenize(text: str) -> List[str]:
    cleaned_text = clean_text(text)
    if not cleaned_text:
        return []
    return cleaned_text.lower().split()


def build_vocab(texts: List[str], min_freq: int = 1) -> Dict[str, int]:
    counter = Counter()
    for text in texts:
        tokens = clean_and_tokenize(text)
        counter.update(tokens)
    vocab = {"<pad>": 0, "<unk>": 1}
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    return vocab


def calculate_confusion_matrix_for_emotion(
    labels: List[int], predictions: List[int], normalize: str = "true"
) -> pd.DataFrame:
    con_matrix = confusion_matrix(labels, predictions, normalize=normalize).tolist()
    con_matrix_df = pd.DataFrame(
        con_matrix,
        index=[str(i) for i in range(len(con_matrix))],  # True labels
        columns=[str(i) for i in range(len(con_matrix[0]))],  # Predicted labels
    )
    current_matrix_index = con_matrix_df.index
    new_matrix_index = [
        ChatBotConfig.EMOTION_MAPPING[int(i)] for i in current_matrix_index
    ]
    current_matrix_header = con_matrix_df.columns
    new_matrix_header = [
        ChatBotConfig.EMOTION_MAPPING[int(i)] for i in current_matrix_header
    ]

    con_matrix_df.index = new_matrix_index
    con_matrix_df.columns = new_matrix_header

    con_matrix_df = round_all_values_in_df(con_matrix_df, decimals=4)

    con_matrix_df.to_csv(
        f"{TRAIN_LOG_PATH}/confusion_matrix_emotion.csv", index=False, header=True
    )

    return con_matrix_df


def calculate_confusion_matrix_for_topic(
    labels: List[int], predictions: List[int], normalize: str = "true"
) -> pd.DataFrame:
    con_matrix = confusion_matrix(labels, predictions, normalize=normalize).tolist()
    con_matrix_df = pd.DataFrame(
        con_matrix,
        index=[str(i) for i in range(len(con_matrix))],  # True labels
        columns=[str(i) for i in range(len(con_matrix[0]))],  # Predicted labels
    )
    current_matrix_index = con_matrix_df.index
    new_matrix_index = [
        ChatBotConfig.DOMAIN_NUMBER_MAPPING[int(i)] for i in current_matrix_index
    ]
    current_matrix_header = con_matrix_df.columns
    new_matrix_header = [
        ChatBotConfig.DOMAIN_NUMBER_MAPPING[int(i)] for i in current_matrix_header
    ]

    con_matrix_df.index = new_matrix_index
    con_matrix_df.columns = new_matrix_header

    con_matrix_df = round_all_values_in_df(con_matrix_df, decimals=4)

    con_matrix_df.to_csv(
        f"{TRAIN_LOG_PATH}/confusion_matrix_topic.csv", index=False, header=True
    )

    return con_matrix_df


def round_all_values_in_df(df: pd.DataFrame, decimals: int = 2) -> pd.DataFrame:
    cols = df.columns
    for col in cols:
        df[col] = df[col].apply(lambda x: round(x, decimals))
    return df


def sanitize_text(text: str) -> str:
    return (
        text.replace("’", "'")
        .replace("‘", "'")
        .replace("“", '"')
        .replace("”", '"')
        .replace("–", "-")
        .replace("—", "-")
        .replace("…", "...")
        .encode("latin-1", "ignore")
        .decode("latin-1")
    )
