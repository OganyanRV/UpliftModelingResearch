import os
import json
import time
import torch
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Union, Optional, Tuple
from torch.utils.data import DataLoader, TensorDataset as TorchDataset
import torch.nn as nn
import torch.nn.functional as F
from src.models.NNUpliftModeling.INNUpliftModeling import INNUpliftModeling
from src.models.NNUpliftModeling.EFIN import *


class EFINUpliftModel(INNUpliftModeling):
    """
    EFIN (Explicit Feature Interaction Network) для аплифт-моделирования.
    """
    
    def _initialize_model(self):
        input_dim = self.config.get('input_dim')
        hc_dim = self.config.get('hc_dim', 128)
        hu_dim = self.config.get('hu_dim', 64)
        act_type = self.config.get('act_type', 'elu')
        
        if input_dim is None:
            raise ValueError("input_dim must be specified in the config")

        self.model = EFIN(
            input_dim=input_dim,
            hc_dim=hc_dim,
            hu_dim=hu_dim,
            act_type=act_type
        ).to(self.device)
    
    def _compute_loss(self, outputs, outcome, treatment):
        y_true = outcome.float()
        t_true = treatment.float()
        
        uc = outputs['uc']
        ut = outputs['ut']
        t_logit = outputs['t_logit']
        
        y_true = y_true.unsqueeze(1)
        t_true = t_true.unsqueeze(1)
        
        temp = torch.square((1 - t_true) * uc + t_true * ut - y_true)
        loss1 = torch.mean(temp)
        
        criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
        loss2 = criterion(t_logit, 1 - t_true)

        loss = loss1 + loss2
        
        return loss
    
    @staticmethod
    def generate_config(count, **params):
        """
        Генерация конфигураций для EFIN модели.
        
        Args:
            count: количество конфигураций
            **params: дополнительные параметры
            
        Returns:
            Список конфигураций
        """
        # Базовые параметры для EFIN
        efin_params = {
            'input_dim': 32,              # Размерность входных данных
            'hc_dim': [64, 128, 256],     # Варианты размерности Control Net
            'hu_dim': [32, 64, 128],      # Варианты размерности Uplift Net
            'is_self': [True, False],     # Использовать ли дополнительный слой
            'act_type': ['elu', 'relu', 'sigmoid', 'tanh'],  # Функции активации
            'batch_size': [32, 64, 128],  # Размеры батчей
            'learning_rate': [0.001, 0.01]  # Скорости обучения
        }
        
        # Объединение с переданными параметрами
        for key, value in params.items():
            efin_params[key] = value
        
        # Генерация конфигураций с использованием базового метода
        return INNUpliftModeling.generate_config(count, efin_params)
