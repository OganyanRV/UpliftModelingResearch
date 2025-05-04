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
from src.models.NNUpliftModeling.DESCN import *


class DESCNUpliftModel(INNUpliftModeling):
    """
    DESCN для аплифт-моделирования.
    """
    
    def _initialize_model(self):
        input_dim = self.config.get('input_dim')
        share_dim = self.config.get('share_dim', 128)
        base_dim = self.config.get('base_dim', 64)
        do_rate = self.config.get('do_rate', 0.2)
        batch_norm = self.config.get('batch_norm', False)
        normalization = self.config.get('normalization', 'none')
        
        if input_dim is None:
            raise ValueError("input_dim must be specified in the config")

        self.model = DESCN(
            input_dim=input_dim,
            share_dim=share_dim,
            base_dim=base_dim,
            do_rate=do_rate,
            device=self.device,
            batch_norm=batch_norm,
            normalization=normalization
        )
    
    def _compute_loss(self, outputs, outcome, treatment):
        """
        Вычисление функции потерь для DESCN.
        
        Args:
            outputs: выход модели
            outcome: целевая переменная
            treatment: индикатор воздействия
            
        Returns:
            Значение функции потерь
        """
        # Извлечение необходимых выходов модели
        mu1_logit = outputs['mu1_logit']
        mu0_logit = outputs['mu0_logit']
        p_prpsy_logit = outputs['p_prpsy_logit']
        
        # Веса для разных компонентов потери
        factual_loss_weight = self.config.get('factual_loss_weight', 1.0)
        propensity_loss_weight = self.config.get('propensity_loss_weight', 0.1)
        tau_loss_weight = self.config.get('tau_loss_weight', 0.1)
        
        # Формируем маски для групп воздействия и контроля
        treatment_mask = (treatment == 1).float().unsqueeze(1)
        control_mask = (treatment == 0).float().unsqueeze(1)
        
        # Фактическая потеря - MSE для фактических наблюдений
        y_pred = treatment_mask * mu1_logit + control_mask * mu0_logit
        factual_loss = F.mse_loss(y_pred, outcome.unsqueeze(1))
        
        # Потеря для предсказания вероятности назначения воздействия
        propensity_loss = F.binary_cross_entropy_with_logits(
            p_prpsy_logit.squeeze(), 
            treatment
        )
        
        # Потеря для предсказания эффекта воздействия (если известно)
        if self.config.get('use_tau_loss', False) and hasattr(self, 'tau_true'):
            tau_loss = F.mse_loss(outputs['tau_logit'], self.tau_true)
        else:
            tau_loss = torch.tensor(0.0, device=self.device)
        
        # Общая потеря
        total_loss = (
            factual_loss_weight * factual_loss + 
            propensity_loss_weight * propensity_loss + 
            tau_loss_weight * tau_loss
        )
        
        return total_loss
    
    def _process_prediction_outputs(self, outputs):
        """
        Обработка выходов модели для предсказания.
        
        Args:
            outputs: выходы модели
            
        Returns:
            Словарь с предсказанными значениями
        """
        # Выделяем и преобразуем нужные для предсказания поля
        return {
            'y0': outputs['mu0_logit'],
            'y1': outputs['mu1_logit'],
            'uplift': outputs['uplift'],
            'propensity': outputs['p_prpsy']
        }
    
    @staticmethod
    def generate_config(count, params):
        """
        Генерация конфигураций для DESCN модели.
        
        Args:
            count: количество конфигураций
            **params: дополнительные параметры
            
        Returns:
            Список конфигураций
        """
        # Базовые параметры для DESCN
        descn_params = {
            'input_dim': 100,             # Должно быть задано в соответствии с данными
            'share_dim': [256, 256], # Варианты размерности общих слоев
            'base_dim': [256],   # Варианты размерности базовых слоев
            'do_rate': [0.1, 0.2, 0.3],  # Варианты dropout
            'batch_norm': [True, False], # Использование BatchNorm
            'normalization': ['none', 'divide'], # Тип нормализации
            'factual_loss_weight': [0.8, 1.0, 1.2], # Вес фактической потери
            'propensity_loss_weight': [0.05, 0.1, 0.2], # Вес потери пропенсити
            'tau_loss_weight': [0.05, 0.1, 0.2],    # Вес потери tau (если применимо)
            'gradient_accumulation_steps' : 2
        }
        
        # Объединение с переданными параметрами
        for key, value in params.items():
            descn_params[key] = value
        
        # Генерация конфигураций с использованием базового метода
        return INNUpliftModeling.generate_config(count, descn_params)

    def num_params(self):
        return sum([p.numel() for p in self.model.parameters() if p.requires_grad])