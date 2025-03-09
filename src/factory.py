from src.models.CausalML.Models import TModel, SModel, XModel, DRModel, UpliftRandomForestModel
from src.models.ScikitML.Models import ClassTransformationModel, ClassTransformationRegModel
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

class ClassTransformationFactory(IFactory):
    @staticmethod
    def create(config_json, train_path, test_path):
        model = ClassTransformationModel(config_json)
        train = NumpyDataset(train_path)
        test = NumpyDataset(test_path)
        return model, train, test

class ClassTransformationRegFactory(IFactory):
    @staticmethod
    def create(config_json, train_path, test_path):
        model = ClassTransformationRegModel(config_json)
        train = NumpyDataset(train_path)
        test = NumpyDataset(test_path)
        return model, train, test

class UpliftRandomForestModelFactory(IFactory):
    @staticmethod
    def create(config_json, train_path, test_path):
        model = UpliftRandomForestModel(config_json)
        train = NumpyDataset(train_path)
        test = NumpyDataset(test_path)
        return model, train, test