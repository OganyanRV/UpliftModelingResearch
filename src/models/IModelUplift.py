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
        self.model = None
        self.config = None

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