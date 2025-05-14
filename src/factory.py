from src.models.CausalML.Models import TModel, SModel, XModel, DRModel, UpliftRandomForestModel
from src.models.ScikitML.Models import ClassTransformationModel, ClassTransformationRegModel
from src.models.NNUpliftModeling.DESCNUpliftModel import DESCNUpliftModel
from src.models.NNUpliftModeling.DESCNDistillation import DESCNDistillation
from src.models.NNUpliftModeling.EFINUpliftModel import EFINUpliftModel

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

class ClassTransformationModelFactory(IFactory):
    @staticmethod
    def create(config_json, train_path, test_path):
        model = ClassTransformationModel(config_json)
        train = NumpyDataset(train_path)
        test = NumpyDataset(test_path)
        return model, train, test

class ClassTransformationRegModelFactory(IFactory):
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

class DESCNUpliftModelFactory(IFactory):
    @staticmethod
    def create(config_json, train_path, test_path):
        model = DESCNUpliftModel(config_json)
        train = TorchDataset(train_path)
        test = TorchDataset(test_path)
        return model, train, test

class DESCNDistillationFactory(IFactory):
    @staticmethod
    def create(config_json, train_path, test_path):
        model = DESCNDistillation(config_json)
        train = TorchDataset(train_path)
        test = TorchDataset(test_path)
        return model, train, test

class EFINUpliftModelFactory(IFactory):
    @staticmethod
    def create(config_json, train_path, test_path):
        model = EFINUpliftModel(config_json)
        train = TorchDataset(train_path)
        test = TorchDataset(test_path)
        return model, train, test