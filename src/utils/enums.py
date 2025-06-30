import torch
from pathlib import Path


class ChatBotConfig:
    DEFAULT_CHAT_MODEL = "gpt-4o-mini"
    GPT_4O = "gpt-4o"
    DEFAULT_PROMPT_PATH = "src/prompts/base_prompt.yml"
    COT_PROMPT_PATH = "src/prompts/chain_of_thought_prompt.yml"
    N_HISTORY_CHATS = 10
    QUERY_ENGINE_TOOL = "query_engine"
    QUERY_ENGINE_DESCRIPTION = "Query engine for the index"
    DOMAINS = [
        "Cancer",
        "Diabetes",
        "Disease Control and Prevention",
        "Genetic and Rare Diseases",
        "Growth Hormone Receptor",
        "Heart, Lung and Blood",
        "Neurological Disorders and Stroke",
        "Senior Health",
        "Others",
    ]
    CANCER = "cancer"
    DIABETES = "diabetes"
    DISEASE_CONTROL_AND_PREVENTION = "disease_control_and_prevention"
    GENETIC_AND_RARE_DISEASES = "genetic_and_rare_diseases"
    GROWTH_HORMONE_RECEPTOR = "growth_hormone_receptor"
    HEART_LUNG_AND_BLOOD = "heart_lung_and_blood"
    NEUROLOGICAL_DISORDERS_AND_STROKE = "neurological_disorders_and_stroke"
    SENIOR_HEALTH = "senior_health"
    OTHERS = "others"
    DOMAIN_MAPPING = {
        "Cancer": CANCER,
        "Diabetes": DIABETES,
        "Disease Control and Prevention": DISEASE_CONTROL_AND_PREVENTION,
        "Genetic and Rare Diseases": GENETIC_AND_RARE_DISEASES,
        "Growth Hormone Receptor": GROWTH_HORMONE_RECEPTOR,
        "Heart, Lung and Blood": HEART_LUNG_AND_BLOOD,
        "Neurological Disorders and Stroke": NEUROLOGICAL_DISORDERS_AND_STROKE,
        "Senior Health": SENIOR_HEALTH,
        "Others": OTHERS,
    }
    EMOTION_MAPPING = {
        0: "sadness",
        1: "joy",
        2: "love",
        3: "anger",
        4: "fear",
        5: "surprise",
    }


class IngestionConfig:
    DATA_PATH = "src/data/medical_data"
    INDEX_PATH = "src/indices"
    COL_NAME_TO_INGEST = "Answer"
    CSV_FILE_EXTENSION = ".csv"


class EmotionRecognitionConfig:
    LABEL_COL = "label"
    TEXT_COL = "text"
    FLOAT32 = "float32"
    LONG = "long"
    LSTM_MODEL_PATH = "src/ml_models/model_files/lstm_model.pth"
    CNN_MODEL_PATH = "src/ml_models/model_files/cnn_model.pth"
    TRAIN_DATA_PATH = "src/data/emotion_data/training.csv"
    TEST_DATA_PATH = "src/data/emotion_data/test.csv"
    VALIDATION_DATA_PATH = "src/data/emotion_data/validation.csv"
    MAX_SEQ_LENGTH = 100
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DEFAULT_EMBED_DIM = 100
    DEFAULT_NUM_CLASSES = 6
    DEFAULT_KERNEL_SIZES = [3, 4, 5]
    DEFAULT_NUM_FILTERS = 100
    DEFAULT_DROPOUT = 0.2
    DEFAULT_LR = 0.001

class QuestionRecommendConfig:
    BASE_DIR = Path(__file__).resolve().parent.parent
    print(f"BASE_DIR: {BASE_DIR}")
    FINE_TUNE_DATA_DIR = BASE_DIR/"data"/"fine_tune_dataset"/"CancerQA.csv"
    PROCESSED_DATA_DIR = BASE_DIR/"data"/"processed"
    MODEL_DIR = BASE_DIR/"data"/"processed"