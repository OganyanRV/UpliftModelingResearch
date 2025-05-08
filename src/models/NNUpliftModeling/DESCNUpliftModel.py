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
        """
        # Распаковка необходимых выходов модели
        p_prpsy_logit = outputs['p_prpsy_logit']
        p_estr = outputs['p_estr']
        p_escr = outputs['p_escr']
        p_tau_logit = outputs['tau_logit']
        p_mu1_logit = outputs['mu1_logit']
        p_mu0_logit = outputs['mu0_logit']
        shared_h = outputs['shared_h']
        
        # Веса для разных компонентов потери из конфигурации
        prpsy_w = self.config.get('prpsy_w', 1.0)
        escvr1_w = self.config.get('escvr1_w', 1.0)
        escvr0_w = self.config.get('escvr0_w', 1.0)
        mu1hat_w = self.config.get('mu1hat_w', 1.0)
        mu0hat_w = self.config.get('mu0hat_w', 1.0)
        
        # Преобразование целевой переменной и индикатора воздействия
        y_labels = outcome.unsqueeze(1).float()
        t_labels = treatment.unsqueeze(1).float()
        
        # Маски для групп воздействия и контроля
        treatment_mask = t_labels.bool()
        control_mask = ~treatment_mask
        
        loss_fn = nn.BCELoss()
        loss_with_logit_fn = nn.BCEWithLogitsLoss()
        
        prpsy_loss = prpsy_w * loss_with_logit_fn(p_prpsy_logit, t_labels)
        
        estr_loss = escvr1_w * loss_fn(p_estr, y_labels * t_labels)
        escr_loss = escvr0_w * loss_fn(p_escr, y_labels * (1 - t_labels))

        cross_tr_loss = mu1hat_w * loss_fn(
            torch.sigmoid(p_mu0_logit + p_tau_logit)[treatment_mask],
            y_labels[treatment_mask]
        )

        cross_tr_loss = 0.0 if torch.isnan(cross_tr_loss) else cross_tr_loss
        
        cross_cr_loss = mu0hat_w * loss_fn(
            torch.sigmoid(p_mu1_logit - p_tau_logit)[control_mask],
            y_labels[control_mask]
        )

        cross_cr_loss = 0.0 if torch.isnan(cross_cr_loss) else cross_cr_loss

        # print(prpsy_loss, estr_loss, escr_loss, cross_cr_loss, cross_tr_loss)
        total_loss = prpsy_loss + estr_loss + escr_loss + cross_tr_loss + cross_cr_loss
        # print(total_loss)
        if torch.isnan(total_loss):
            print(prpsy_loss, estr_loss, escr_loss, cross_cr_loss, cross_tr_loss)
            print(y_labels, y_labels.shape)
            print(treatment_mask, treatment_mask.shape)
            print(y_labels[treatment_mask])
            print(p_mu0_logit + p_tau_logit)
            print(torch.sigmoid(p_mu0_logit + p_tau_logit)[treatment_mask])
            print(mu1hat_w)
            
            raise
        
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
            'prpsy_w': 0.5,   # Вес для потери пропенсити
            'escvr1_w': 0.5,  # Вес для потери ESTR
            'escvr0_w': 1.0,  # Вес для потери ESCR
            'mu1hat_w': 1.0,  # Вес для перекрестной потери TR
            'mu0hat_w': 0.5,  # Вес для перекрестной потери CR
            'gradient_accumulation_steps' : 2 # Количество шагов для аккумуляции градиентов
        }
        
        # Объединение с переданными параметрами
        for key, value in params.items():
            descn_params[key] = value
        
        # Генерация конфигураций с использованием базового метода
        return INNUpliftModeling.generate_config(count, descn_params)

    def num_params(self):
        return sum([p.numel() for p in self.model.parameters() if p.requires_grad])