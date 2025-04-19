import pandas as pd

from src.core_managers.data_processing_managers.base_data_processing_manager import BaseDataProcessingManager


class EmotionDataProcessingManager(BaseDataProcessingManager):
    def __init__(self, df: pd.DataFrame):
        super().__init__(df)
