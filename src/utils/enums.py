class ChatBotConfig:
    DEFAULT_CHAT_MODEL = "gpt-4o-mini"
    GPT_4O = "gpt-4o"
    DEFAULT_PROMPT_PATH = "src/prompts/base_prompt.yml"

class IngestionConfig:
    DATA_PATH = "src/data/cancer"
    INDEX_PATH = "src/indices"
    COL_NAME_TO_INGEST = "Answer"
    CSV_FILE_EXTENSION = ".csv"


class LSTMConfig:
    LABEL_COL = "label"
    TEXT_COL = "text"
    FLOAT32 = "float32"
    LONG = "long"
    MODEL_PATH = "src/ml_models/model_files/lstm_model.pth"
