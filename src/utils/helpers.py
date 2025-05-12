import hashlib
from typing import Dict, List
from uuid import UUID, uuid4

import yaml


def load_yml_configs(config_path: str) -> Dict:
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def get_unique_id() -> UUID:
    return uuid4()


def hash_string(string: str) -> str:
    return hashlib.sha256(string.encode()).hexdigest()


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
