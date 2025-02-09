import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from abc import ABC, abstractmethod

class IDataset(ABC):
    def __init__(self):
        self.data = None

class TorchDataset(IDataset, Dataset):
    def __init__(self, path):
        IDataset.__init__(self)
        Dataset.__init__(self)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pandas = pd.read_csv(path, sep='\t')
        self.data = torch.tensor(self.pandas.drop(['treatment', 'target'], axis=1).values, dtype=torch.float32).to(device)
        self.target = torch.tensor(self.pandas['target'].values, dtype=torch.float32).to(device)
        self.treatment = torch.tensor(self.pandas['treatment'].values, dtype=torch.float32).to(device)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx], self.treatment[idx]

class NumpyDataset(IDataset):
    def __init__(self, path=None, from_dataset=False, dataset=None):
        IDataset.__init__(self)
        if from_dataset == False:
            self.data = pd.read_csv(path, sep='\t')
        else:
            self.data = dataset
        self.col_treatment = "treatment"
        self.col_target = "target"
        self.cols_features = self.data.drop(["treatment", 'target'], axis=1).columns

    def __getitem__(self, index):
        if isinstance(index, slice):
            return NumpyDataset(from_dataset=True, dataset=self.data.iloc[index])
        elif isinstance(index, int):
            if index < 0:
                index += len(self.data)
            if 0 <= index < len(self.data):
                return self.data.iloc[index]
            else:
                raise IndexError(f"Индекс {index} выходит за пределы границ data")
        else:
            raise TypeError("Индекс должен быть int или slice")

    def __len__(self):
        return len(self.data)


def sample_features(percents, train_data, test_data, output_dir):
    """
    Оставляет в датасете случайные percent из percents процентов фичей, иммитируя достпуность бОльшего количества фичей в рантайме
    """
    if train_data.shape[1] != test_data.shape[1]:
        raise ValueError("Тренировочный и тестовый датасеты имеют разное количество фичей!")

    os.makedirs(output_dir, exist_ok=True)
    
    for percent in percents:
        n_features = train_data.shape[1]  # Количество колонок (фичей)
        num_sampled_features = int(n_features * percent / 100)  # Количество фичей для выборки    
        sampled_features = np.sort(np.random.permutation(train_data.drop(['treatment', 'target'], axis=1).columns)[:num_sampled_features])    
        train_sampled = train_data.loc[:, [*sampled_features, 'treatment', 'target']]
        test_sampled = test_data.loc[:, [*sampled_features, 'treatment', 'target']]

        os.makedirs(os.path.join(output_dir, str(percent)), exist_ok=True)
        train_out_path = os.path.join(output_dir, str(percent), 'train.tsv')
        test_out_path = os.path.join(output_dir, str(percent), 'test.tsv')
        pd.DataFrame(train_sampled).to_csv(train_out_path, sep='\t', index=False)
        pd.DataFrame(test_sampled).to_csv(test_out_path, sep='\t', index=False)