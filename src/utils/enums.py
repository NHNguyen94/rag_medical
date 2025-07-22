from pathlib import Path

import torch


class GeneralConfig:
    DEVICE = torch.device("mps" if torch.cuda.is_available() else "cpu")
    DEFAULT_LANGUAGE = "en"
    FR_LANGUAGE = "fr"


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
    DOMAIN_ENCODE_MAPPING = {
        CANCER: 0,
        DIABETES: 1,
        DISEASE_CONTROL_AND_PREVENTION: 2,
        GENETIC_AND_RARE_DISEASES: 3,
        GROWTH_HORMONE_RECEPTOR: 4,
        HEART_LUNG_AND_BLOOD: 5,
        NEUROLOGICAL_DISORDERS_AND_STROKE: 6,
        SENIOR_HEALTH: 7,
        OTHERS: 8,
    }
    DOMAIN_NUMBER_MAPPING = {
        0: CANCER,
        1: DIABETES,
        2: DISEASE_CONTROL_AND_PREVENTION,
        3: GENETIC_AND_RARE_DISEASES,
        4: GROWTH_HORMONE_RECEPTOR,
        5: HEART_LUNG_AND_BLOOD,
        6: NEUROLOGICAL_DISORDERS_AND_STROKE,
        7: SENIOR_HEALTH,
        8: OTHERS,
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
    DEVICE = GeneralConfig.DEVICE
    DEFAULT_EMBED_DIM = 100
    DEFAULT_NUM_CLASSES = 6
    DEFAULT_KERNEL_SIZES = [3, 4, 5]
    DEFAULT_NUM_FILTERS = 100
    DEFAULT_DROPOUT = 0.2
    DEFAULT_LR = 0.001


class QuestionRecommendConfig:
    BASE_DIR = Path(__file__).resolve().parent.parent
    print(f"BASE_DIR: {BASE_DIR}")
    FINE_TUNE_DATA_DIR = BASE_DIR / "data" / "fine_tune_dataset" / "CancerQA.csv"
    PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
    MODEL_DIR = BASE_DIR / "data" / "processed"


class TopicClusteringConfig:
    DEFAULT_MODEL = "google/bert_uncased_L-2_H-128_A-2"
    DEFAULT_DROPOUT = 0.2
    DEFAULT_LR = 0.001
    DEFAULT_NUM_CLASSES = 9
    MAX_SEQ_LENGTH = 148
    DEVICE = GeneralConfig.DEVICE
    LABEL_COL = "topic"
    TEXT_COL = "Question"
    TRAIN_DATA_PATH = "src/data/medical_data/all/training.csv"
    TEST_DATA_PATH = "src/data/medical_data/all/test.csv"
    MODEL_PATH = "src/ml_models/model_files/topic_clustering_model.pth"


class AudioConfig:
    DEVICE = "cpu"  # no support for mps yet
    COMPUTE_TYPE = "float32"  # no support for float16 yet
    RECORDINGS_AUDIO_DIR = "src/data/recordings_from_speaker"
    OUTPUT_TTS_AUDIO_DIR = "src/data/output_tts_audio"
    AUDIO_FILE_EXTENSION = ".wav"
    FRAME_RATE = 44100
    CHANNELS = 1
    WIDTH = 2
    AUDIO_FORMAT = "audio/wav"
    # DEFAULT_WHISPER_MODEL = "large-v2"
    DEFAULT_WHISPER_MODEL = "base"
    DEFAULT_LANGUAGE = "en"


class RedisConfig:
    DEFAULT_EXPIRATION = 600

class AdminConfig:
    CUSTOMIZED_SYSTEM_PROMPT_DIR = "src/data/customized_system_prompt"
