
import os
import pandas as pd
import numpy as np
import torch
from abc import ABC, abstractmethod


class IDataset(ABC):
    def __init__(self):
        self.train_data = None
        self.test_data = None

    @abstractmethod
    def load(self, path):
        """Загружает данные"""
        pass

    @abstractmethod
    def sample_features(self, percent, output_dir):
        """
        Оставляет percent процент фичей и сохраняет результат.
        Аргументы:
        - percent: float — доля фичей (0-1).
        - output_dir: str — путь к папке для сохранения.
        """
        pass


class TorchDataset(IDataset):
    def load(self, path):
        """Загружает данные в формате PyTorch тензоров"""
        train_path = os.path.join(path, 'train.tsv')
        test_path = os.path.join(path, 'test.tsv')

        self.train_data = torch.tensor(pd.read_csv(train_path, sep='\t').values)
        self.test_data = torch.tensor(pd.read_csv(test_path, sep='\t').values)

    def sample_features(self, percent, output_dir):
        os.makedirs(output_dir, exist_ok=True)

        n_features = self.train_data.shape[1]  # Количество колонок (фичей)
        sampled_features = int(n_features * percent)  # Количество фичей для выборки

        sampled_indices = torch.randperm(n_features)[:sampled_features]

        train_sampled = self.train_data[:, sampled_indices]
        test_sampled = self.test_data[:, sampled_indices]

        # Сохраняем результат в файлы
        train_out_path = os.path.join(output_dir, 'train.tsv')
        test_out_path = os.path.join(output_dir, 'test.tsv')

        pd.DataFrame(train_sampled.numpy()).to_csv(train_out_path, sep='\t', index=False, header=False)
        pd.DataFrame(test_sampled.numpy()).to_csv(test_out_path, sep='\t', index=False, header=False)


class NumpyDataset(IDataset):
    def load(self, path):
        """Загружает данные в формате NumPy массивов"""
        train_path = os.path.join(path, 'train.tsv')
        test_path = os.path.join(path, 'test.tsv')

        self.train_data = pd.read_csv(train_path, sep='\t').values
        self.test_data = pd.read_csv(test_path, sep='\t').values

    def sample_features(self, percent, output_dir):
        os.makedirs(output_dir, exist_ok=True)

        n_features = self.train_data.shape[1]  # Количество колонок (фичей)
        sampled_features = int(n_features * percent)  # Количество фичей для выборки

        sampled_indices = np.random.permutation(n_features)[:sampled_features]

        train_sampled = self.train_data[:, sampled_indices]
        test_sampled = self.test_data[:, sampled_indices]

        # Сохраняем результат в файлы
        train_out_path = os.path.join(output_dir, 'train.tsv')
        test_out_path = os.path.join(output_dir, 'test.tsv')

        pd.DataFrame(train_sampled).to_csv(train_out_path, sep='\t', index=False, header=False)
        pd.DataFrame(test_sampled).to_csv(test_out_path, sep='\t', index=False, header=False)


if __name__ == '__main__':
    # Пример загрузки и работы с TorchDataset
    path_to_data = 'path_to_your_dataset_folder'  # Укажите свой путь
    output_directory = 'output_folder'  # Папка для сохранения выборок

    # TorchDataset использование
    torch_dataset = TorchDataset()
    torch_dataset.load(path_to_data)
    torch_dataset.sample_features(0.5, output_directory)

    # NumpyDataset использование
    numpy_dataset = NumpyDataset()
    numpy_dataset.load(path_to_data)
    numpy_dataset.sample_features(0.5, output_directory)