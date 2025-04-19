from abc import ABC, abstractmethod

import pandas as pd


class BaseDataProcessingManager(ABC):
    def __init__(self, df: pd.DataFrame):
        self.df = df

    @abstractmethod
    def process(self):
        pass

    @abstractmethod
    def split_train_test(self):
        pass
