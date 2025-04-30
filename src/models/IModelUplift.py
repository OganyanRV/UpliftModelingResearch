from src.datasets import sample_features, IDataset, TorchDataset, NumpyDataset

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
        """
        Инициализация объекта модели.
        
        Args:
            config_json: строка с JSON-конфигурацией модели
            from_load: флаг, указывающий, что модель загружается из файла
            path: путь для загрузки модели
        """
        self.model = None
        self.config = None

    @abstractmethod
    def fit(self, X: IDataset):
        """
        Метод для обучения модели.
        """
        pass

    @abstractmethod
    def predict(self, X: IDataset):
        """
        Метод для предсказания.
        """
        pass

    @abstractmethod
    def predict_light(self, X: IDataset):
        """
        Метод для предсказания без возвращения предикта.
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

    @staticmethod
    def generate_config(count, **params):
        """
        Статические метод для генерации конфигов, по которым можно собрать модель
        """
        pass