from src.models.ICausalML.Models import TModel, SModel, XModel, DRModel
from abc import ABC, abstractmethod
from src.datasets import TorchDataset, NumpyDataset

class IFactory(ABC):
    @staticmethod
    @abstractmethod
    def create():
        """Фабричный метод, создающий объект модели и датасета."""
        pass

class TModelFactory(IFactory):
    @staticmethod
    def create(config_json, train_path, test_path):
        model = TModel(config_json)
        train = NumpyDataset(train_path)
        test = NumpyDataset(test_path)
        return model, train, test

class SModelFactory(IFactory):
    @staticmethod
    def create(config_json, train_path, test_path):
        model = SModel(config_json)
        train = NumpyDataset(train_path)
        test = NumpyDataset(test_path)
        return model, train, test


class XModelFactory(IFactory):
    @staticmethod
    def create(config_json, train_path, test_path):
        model = XModel(config_json)
        train = NumpyDataset(train_path)
        test = NumpyDataset(test_path)
        return model, train, test

class DRModelFactory(IFactory):
    @staticmethod
    def create(config_json, train_path, test_path):
        model = DRModel(config_json)
        train = NumpyDataset(train_path)
        test = NumpyDataset(test_path)
        return model, train, test