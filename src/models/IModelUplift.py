from src.datasets import sample_features, TorchDataset, NumpyDataset

import json
import os
import pickle
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import copy

import time

class IModelUplift(ABC):
    """
    Интерфейс для реализации моделей uplift.
    """

    def __init__(self, config_json=None, from_load=False, path=None):
        if from_load == False:
            if config_json is None:
                raise ValueError(f"No config while contstructing model.")
            self.model = None
            self.config = config_json
        else:
            if path is None:
                raise ValueError(f"No config or model paths while contstructing model.")
            # Дебильный баг, что если сделать self.moldel=loaded_model то models_t,
            #models_s не будут внутри self.model
            model, config = self.load(path)

            self.model = model
            self.config = config

    @abstractmethod
    def fit(self, X: NumpyDataset):
        """
        Метод для обучения модели.
        """
        pass

    @abstractmethod
    def predict(self, X: NumpyDataset):
        """
        Метод для предсказания.
        """
        pass

    @abstractmethod
    def predict_light(self, X: NumpyDataset):
        """
        Метод для предсказания.
        """
        pass

    @abstractmethod
    def load(self, path):
        """
        Метод для конструирования модели из файла.
        """
        pass

    @abstractmethod
    def measure_inference_time(self, data, batch_size, max_size=None):
        """
        Метод для измерения среднего времени инференса модели на данных.
        """
        pass