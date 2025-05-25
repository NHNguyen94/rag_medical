import hashlib
import re
import textwrap
from typing import Dict, List
from uuid import UUID, uuid4

import nltk
import pandas as pd
import yaml
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def load_yml_configs(config_path: str) -> Dict:
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


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


def write_log_file(log_file_path: str, data: Dict, with_header: bool) -> None:
    df = pd.DataFrame([data])
    df.to_csv(log_file_path, mode="a", index=False, header=with_header)


def download_nlkt() -> None:
    nltk.download("punkt")
    nltk.download("punkt_tab")
    nltk.download("stopwords")


def clean_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\d+", "", text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalpha()]
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words]

    return " ".join(tokens)
