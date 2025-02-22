from src.models.ICausalML.Models import TModel
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