import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from abc import ABC, abstractmethod
import pickle

from src.global_params import COL_TARGET, COL_TREATMENT

class IDataset(ABC):
    def __init__(self):
        self.data = None

class TorchDataset(IDataset, Dataset):
    def __init__(self, path):
        IDataset.__init__(self)
        Dataset.__init__(self)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pandas = pd.read_csv(path, sep='\t')
        self.data = torch.tensor(self.pandas.drop([COL_TREATMENT, COL_TARGET], axis=1).values, dtype=torch.float32).to(device)
        self.target = torch.tensor(self.pandas[COL_TARGET].values, dtype=torch.float32).to(device)
        self.treatment = torch.tensor(self.pandas[COL_TREATMENT].values, dtype=torch.float32).to(device)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.treatment[idx], self.target[idx]

class NumpyDataset(IDataset):
    def __init__(self, path=None, from_dataset=False, dataset=None):
        IDataset.__init__(self)
        if from_dataset == False:
            self.data = pd.read_csv(path, sep='\t')
        else:
            self.data = dataset
        self.col_treatment = COL_TREATMENT
        self.col_target = COL_TARGET
        self.cols_features = self.data.drop([COL_TARGET, COL_TREATMENT], axis=1).columns

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
        sampled_features = np.sort(np.random.permutation(train_data.drop([COL_TREATMENT, COL_TARGET], axis=1).columns)[:num_sampled_features])    
        train_sampled = train_data.loc[:, [*sampled_features, COL_TREATMENT, COL_TARGET]]
        test_sampled = test_data.loc[:, [*sampled_features, COL_TREATMENT, COL_TARGET]]

        os.makedirs(os.path.join(output_dir, str(percent)), exist_ok=True)
        train_out_path = os.path.join(output_dir, str(percent), 'train.tsv')
        test_out_path = os.path.join(output_dir, str(percent), 'test.tsv')
        pd.DataFrame(train_sampled).to_csv(train_out_path, sep='\t', index=False)
        pd.DataFrame(test_sampled).to_csv(test_out_path, sep='\t', index=False)


class PairedUpliftDataset(IDataset, torch.utils.data.Dataset):
    """
    Датасет, содержащий пары примеров (treatment, control) с предсказаниями учителя.
    """
    def __init__(self, teacher_model, path=None, from_saved_path=None):
        """
        Инициализация датасета.
        teacher_model: Предобученная модель-учитель
        """
        IDataset.__init__(self)
        torch.utils.data.Dataset.__init__(self)

        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if from_saved_path:
            self.load(from_saved_path)
        else:
            self.pandas = pd.read_csv(path, sep='\t')
            self.data = torch.tensor(self.pandas.drop([COL_TREATMENT, COL_TARGET], axis=1).values, dtype=torch.float32).to(self.device)
            self.target = torch.tensor(self.pandas[COL_TARGET].values, dtype=torch.float32).to(self.device)
            self.treatment = torch.tensor(self.pandas[COL_TREATMENT].values, dtype=torch.float32).to(self.device)
    
            # Разделяем примеры на группы воздействия и контроля
            treatment_mask = self.treatment == 1
            control_mask = self.treatment == 0
            
            self.treatment_indices = np.where(treatment_mask)[0]
            self.control_indices = np.where(control_mask)[0]
            
            teacher_preds = teacher_model.predict(NumpyDataset(path)) # return p - q
            # p + q == 1
            # p - q == score
            # p = (1+score) / 2
            # q = (1-score) / 2

            def f(x):
                if x.treatment == 1:
                    return (1 + x['score']) / 2
                return (1 - x['score']) / 2
    
            teacher_preds['score2'] = teacher_preds.apply(f, axis=1)
            
            self.teacher_preds = torch.tensor(
                teacher_preds['score2'].values,
                dtype=torch.float32
            )
        
        self.pairs = self._create_pairs()
    
    def _create_pairs(self):
        """
        Создает пары из примеров групп воздействия и контроля.
        """
        # Здесь мы используем случайное сопоставление примеров как в статье
        
        np.random.shuffle(self.treatment_indices)
        np.random.shuffle(self.control_indices)
        
        n_pairs = min(len(self.treatment_indices), len(self.control_indices))
        
        pairs = [
            (self.treatment_indices[i], self.control_indices[i])
            for i in range(n_pairs)
        ]
        
        return pairs
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        
        t_idx, c_idx = self.pairs[idx]
        
        # Извлекаем данные для примера из группы воздействия
        t_features = self.data[t_idx]
        t_treatment = self.treatment[t_idx]
        t_outcome = self.target[t_idx]
        t_teacher_pred = self.teacher_preds[t_idx]
        
        # Извлекаем данные для примера из контрольной группы
        c_features = self.data[c_idx]
        c_treatment = self.treatment[c_idx]
        c_outcome = self.target[c_idx]
        c_teacher_pred = self.teacher_preds[c_idx]
        
        return (t_features.to(self.device), t_treatment.to(self.device), t_outcome.to(self.device), t_teacher_pred.to(self.device),
                c_features.to(self.device), c_treatment.to(self.device), c_outcome.to(self.device), c_teacher_pred.to(self.device))
    
    def shuffle_pairs(self):
        """
        Вызывать перед новой эпохой для увеличения разнообразия пар.
        """
        self.pairs = self._create_pairs()


    def save(self, path):
        """
        Сохраняет датасет в файл.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Подготовка данных для сохранения
        save_data = {
            'data': self.data.cpu().numpy() if isinstance(self.data, torch.Tensor) else self.data,
            'treatment': self.treatment.cpu().numpy() if isinstance(self.treatment, torch.Tensor) else self.treatment,
            'target': self.target.cpu().numpy() if isinstance(self.target, torch.Tensor) else self.target,
            'teacher_preds': self.teacher_preds.cpu().numpy() if isinstance(self.teacher_preds, torch.Tensor) else self.teacher_preds,
            'treatment_indices': self.treatment_indices,
            'control_indices': self.control_indices,
            'pairs': self.pairs
        }

        with open(path, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"Dataset saved to {path}")
    
    def load(self, path):
        """
        Загружает датасет из файла.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset file not found: {path}")
        
        # Загружаем данные из файла
        with open(path, 'rb') as f:
            load_data = pickle.load(f)
        
        # Восстанавливаем атрибуты
        self.data = torch.tensor(load_data['data'], dtype=torch.float32)
        self.treatment = torch.tensor(load_data['treatment'], dtype=torch.float32)
        self.target = torch.tensor(load_data['target'], dtype=torch.float32)
        self.teacher_preds = torch.tensor(load_data['teacher_preds'], dtype=torch.float32)
        self.treatment_indices = load_data['treatment_indices']
        self.control_indices = load_data['control_indices']
        self.pairs = load_data['pairs']
        
        print(f"Dataset loaded from {path}")
