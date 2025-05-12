class ChatBotConfig:
    DEFAULT_CHAT_MODEL = "gpt-4o-mini"
    GPT_4O = "gpt-4o"
    DEFAULT_PROMPT_PATH = "src/prompts/base_prompt.yml"
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


class IngestionConfig:
    DATA_PATH = "src/data/medical_data"
    INDEX_PATH = "src/indices"
    COL_NAME_TO_INGEST = "Answer"
    CSV_FILE_EXTENSION = ".csv"


class LSTMConfig:
    LABEL_COL = "label"
    TEXT_COL = "text"
    FLOAT32 = "float32"
    LONG = "long"
    MODEL_PATH = "src/ml_models/model_files/lstm_model.pth"
