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

class DESCNDistillation(INNUpliftModeling):
    """
    DESCN для аплифт-моделирования.
    Soft target + FITNET 6 слойной модели к 3 слойной
    """    
    def __init__(self, config_json=None, from_load=False, path=None, teacher_model=None):
        """
            config_json: строка с JSON-конфигурацией модели ученика
            from_load: флаг, указывающий, что модель загружается из файла
            path: путь для загрузки модели
            teacher_model: учительская модель для дистилляции (DESCNUpliftModel)
        """
        super().__init__(config_json, from_load, path)
        
        self.teacher_model = teacher_model
        
        # Настройки дистилляции из конфигурации
        self.distill_config = self.config.get('distillation', {})
        self.use_response_distill = self.distill_config.get('use_response_distill', True)
        self.use_feature_distill = self.distill_config.get('use_feature_distill', False)
        self.response_weight = self.distill_config.get('response_weight', 0.5)
        self.feature_weight = self.distill_config.get('feature_weight', 0.5)
        self.temperature = self.distill_config.get('temperature', 3.0)
        
        # Создаем адаптеры для feature-based дистилляции
        if self.use_feature_distill and self.teacher_model is not None:
            self._init_feature_adapters()
            
        if self.teacher_model:
            self.teacher_model.model.return_intermediates = self.use_feature_distill
            self.teacher_model.model.eval()
    
    def _initialize_model(self):
        """
        Инициализация модели ученика.
        """
        
        input_dim = self.config.get('input_dim')
        share_dim = self.config.get('share_dim', 64)
        base_dim = self.config.get('base_dim', 32)
        do_rate = self.config.get('do_rate', 0.2)
        batch_norm = self.config.get('batch_norm', False)
        normalization = self.config.get('normalization', 'none')
        descn_version = self.config.get('descn_version', '3')  # Ученик обычно 3-слойный
        
        if input_dim is None:
            raise ValueError("input_dim must be specified in the config")

        self.model = DESCN(
            input_dim=input_dim,
            share_dim=share_dim,
            base_dim=base_dim,
            do_rate=do_rate,
            device=self.device,
            batch_norm=batch_norm,
            normalization=normalization,
            descn_version=descn_version,
            return_intermediates=self.use_feature_distill
        )
    
    def _init_feature_adapters(self):
        """
        Инициализация адаптеров для feature-based дистилляции.
        """
        teacher_share_dim = self.teacher_model.config.get('share_dim', 256)
        student_share_dim = self.config.get('share_dim', 64)
        teacher_base_dim = self.teacher_model.config.get('base_dim', 128)
        student_base_dim = self.config.get('base_dim', 32)
        
        self.layer_adapters = {}
        
        # Адаптеры для основных промежуточных слоев
        # Для 3-слойной модели у нас 3 линейных слоя (0, 1, 2)
        # Для 6-слойной модели у нас 6 линейных слоев (0, 1, 2, 3, 4, 5)
        # Нам нужно отобразить слои ученика на соответствующие слои учителя
        
        # Отображение слоев: ученик -> учитель
        # Слой 0 -> Слой 0
        # Слой 1 -> Слой 2
        # Слой 2 -> Слой 5
        layer_mapping = {0: 0, 1: 2, 2: 5}
        
        for student_idx, teacher_idx in layer_mapping.items():
            if student_idx < 2 and teacher_idx < 5:
                if student_share_dim != teacher_share_dim:
                    self.layer_adapters[f'layer_{student_idx}'] = nn.Linear(
                        student_share_dim, teacher_share_dim
                    ).to(self.device)
                else:
                    self.layer_adapters[f'layer_{student_idx}'] = nn.Identity().to(self.device)
            else:  # Последний слой
                if student_base_dim != teacher_base_dim:
                    self.layer_adapters[f'layer_{student_idx}'] = nn.Linear(
                        student_base_dim, teacher_base_dim
                    ).to(self.device)
                else:
                    self.layer_adapters[f'layer_{student_idx}'] = nn.Identity().to(self.device)
    
    def fit(self, X_train: TorchDataset):
        """
        Обучение модели с дистилляцией знаний.
        """
        if self.teacher_model is None:
            print("No teacher model provided. Training without distillation.")
            return super().fit(X_train)
        
        train_size = int(0.8 * len(X_train))
        val_size = len(X_train) - train_size
        X_train, X_val = torch.utils.data.random_split(X_train, [train_size, val_size])
        
        # Настройка параметров обучения
        epochs = self.config.get('epochs', 10)
        batch_size = self.config.get('batch_size', 32)
        early_stopping_patience = self.config.get('early_stopping_patience', 2)
        accumulation_steps = self.config.get('gradient_accumulation_steps', 1)
        
        train_loader = self._prepare_data_loader(X_train, batch_size, shuffle=True)
        val_loader = self._prepare_data_loader(X_val, batch_size, shuffle=False)
        
        best_val_loss = float('inf')
        best_val_auuc = float('-inf')
        early_stopping_criterion = self.config.get('early_stopping_criterion', 'loss')
        patience_counter = 0
        
        # История обучения
        history = {
            'epoch': [],
            'train_loss': [],
            'train_task_loss': [],
            'train_response_distill_loss': [],
            'train_feature_distill_loss': [],
            'val_loss': [],
            'val_auuc': [],
            'learning_rate': []
        }
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_task_loss = 0.0
            epoch_response_distill_loss = 0.0
            epoch_feature_distill_loss = 0.0
            num_batches = 0
            
            self.model.train()

            for batch_idx, batch in enumerate(train_loader):
                features, treatment, outcome = batch
                
                with torch.no_grad():
                    teacher_outputs = self.teacher_model.model(features)
                
                self.optimizer.zero_grad()
                student_outputs = self.model(features)
                
                # DESCN loss
                task_loss = self._compute_task_loss(student_outputs, outcome, treatment)
                
                response_distill_loss = torch.tensor(0.0).to(self.device)
                feature_distill_loss = torch.tensor(0.0).to(self.device)
                
                if self.use_response_distill:
                    response_distill_loss = self._compute_response_distill_loss(
                        student_outputs, teacher_outputs
                    )

                if self.use_feature_distill:
                    feature_distill_loss = self._compute_feature_distill_loss(
                        student_outputs, teacher_outputs
                    )

                total_loss = task_loss
                if self.use_response_distill:
                    total_loss += self.response_weight * response_distill_loss
                if self.use_feature_distill:
                    total_loss += self.feature_weight * feature_distill_loss

                normalized_loss = total_loss / accumulation_steps
                normalized_loss.backward()
                
                epoch_loss += total_loss.item()
                epoch_task_loss += task_loss.item()
                if self.use_response_distill:
                    epoch_response_distill_loss += response_distill_loss.item()
                if self.use_feature_distill:
                    epoch_feature_distill_loss += feature_distill_loss.item()
                num_batches += 1
                
                if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                    if (batch_idx + 1) % max(1, len(train_loader) // 10) == 0:
                        progress = (batch_idx + 1) / len(train_loader) * 100
                        current_loss = epoch_loss / num_batches
                        print(f"Epoch {epoch+1}/{epochs} - {progress:.1f}% - "
                              f"Loss: {current_loss:.4f}")
            
            avg_train_loss = epoch_loss / num_batches
            avg_task_loss = epoch_task_loss / num_batches
            avg_response_distill_loss = epoch_response_distill_loss / num_batches if self.use_response_distill else 0.0
            avg_feature_distill_loss = epoch_feature_distill_loss / num_batches if self.use_feature_distill else 0.0
            
            # ---- validation ----
            print(f"Validation after epoch")
            val_loss, val_auuc = self._evaluate(val_loader)

            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss if X_val is not None else avg_train_loss)
                else:
                    self.scheduler.step()
            
            history['epoch'].append(epoch + 1)
            history['train_loss'].append(avg_train_loss)
            history['train_task_loss'].append(avg_task_loss)
            history['train_response_distill_loss'].append(avg_response_distill_loss)
            history['train_feature_distill_loss'].append(avg_feature_distill_loss)
            history['val_loss'].append(val_loss)
            history['val_auuc'].append(val_auuc)
            history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, "
                  f"Task: {avg_task_loss:.4f}, Response KD: {avg_response_distill_loss:.4f}, "
                  f"Feature KD: {avg_feature_distill_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val AUUC: {val_auuc:.4f}")
            
            if X_val is not None:
                improved = False
                if early_stopping_criterion == 'loss' and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    improved = True
                    print(f"Validation loss improved to {best_val_loss:.4f}")
                elif early_stopping_criterion == 'auuc' and val_auuc > best_val_auuc:
                    best_val_auuc = val_auuc
                    improved = True
                    print(f"Validation AUUC improved to {best_val_auuc:.4f}")
                    
                if improved:
                    patience_counter = 0
                    best_model_state = {name: param.clone() for name, param in self.model.state_dict().items()}
                    print("Saved best model checkpoint")
                else:
                    patience_counter += 1
                    print(f"No improvement. Patience: {patience_counter}/{early_stopping_patience}")
                    if patience_counter >= early_stopping_patience:
                        print(f"Early stopping triggered after {epoch+1} epochs")
                        # Восстановление лучшей модели
                        self.model.load_state_dict(best_model_state)
                        print("Restored best model")
                        break
        
        if X_val is not None and 'best_model_state' in locals():
            self.model.load_state_dict(best_model_state)
            print("Final model set to best checkpoint")
        
        self.history = history
        return history
    
    def _compute_task_loss(self, outputs, outcome, treatment):
        """
        DESCNN loss
        """
        p_prpsy_logit = outputs['p_prpsy_logit']
        p_estr = outputs['p_estr']
        p_escr = outputs['p_escr']
        p_mu1_logit = outputs['mu1_logit']
        p_mu0_logit = outputs['mu0_logit']
        
        y_labels = outcome.unsqueeze(1).float()
        t_labels = treatment.unsqueeze(1).float()
    
        treatment_mask = t_labels.bool()
        control_mask = ~treatment_mask
        
        loss_fn = nn.BCELoss()
        loss_with_logit_fn = nn.BCEWithLogitsLoss()
        
        prpsy_w = self.config.get('prpsy_w', 1.0)
        escvr1_w = self.config.get('escvr1_w', 1.0)
        escvr0_w = self.config.get('escvr0_w', 1.0)
        mu1hat_w = self.config.get('mu1hat_w', 1.0)
        mu0hat_w = self.config.get('mu0hat_w', 1.0)
        
        prpsy_loss = prpsy_w * loss_with_logit_fn(p_prpsy_logit, t_labels)
        estr_loss = escvr1_w * loss_fn(p_estr, y_labels * t_labels)
        escr_loss = escvr0_w * loss_fn(p_escr, y_labels * (1 - t_labels))
        
        p_tau_logit = outputs['tau_logit']
        
        cross_tr_loss = mu1hat_w * loss_fn(
            torch.sigmoid(p_mu0_logit + p_tau_logit)[treatment_mask],
            y_labels[treatment_mask]
        ) if torch.any(treatment_mask) else 0.0
        
        cross_cr_loss = mu0hat_w * loss_fn(
            torch.sigmoid(p_mu1_logit - p_tau_logit)[control_mask],
            y_labels[control_mask]
        ) if torch.any(control_mask) else 0.0

        total_loss = prpsy_loss + estr_loss + escr_loss + cross_tr_loss + cross_cr_loss
        
        return total_loss
    
    def _compute_response_distill_loss(self, student_outputs, teacher_outputs):
        """
        Response-based distillation.
        """
        student_mu1 = student_outputs['mu1_logit']
        student_mu0 = student_outputs['mu0_logit']
        student_prpsy = student_outputs['p_prpsy_logit']
        
        teacher_mu1 = teacher_outputs['mu1_logit']
        teacher_mu0 = teacher_outputs['mu0_logit']
        teacher_prpsy = teacher_outputs['p_prpsy_logit']
        
        temp = self.temperature

        kl_mu1 = F.kl_div(
            F.log_softmax(student_mu1 / temp, dim=1),
            F.softmax(teacher_mu1 / temp, dim=1),
            reduction='batchmean'
        ) * (temp ** 2)
        
        kl_mu0 = F.kl_div(
            F.log_softmax(student_mu0 / temp, dim=1),
            F.softmax(teacher_mu0 / temp, dim=1),
            reduction='batchmean'
        ) * (temp ** 2)
        
        kl_prpsy = F.kl_div(
            F.log_softmax(student_prpsy / temp, dim=1),
            F.softmax(teacher_prpsy / temp, dim=1),
            reduction='batchmean'
        ) * (temp ** 2)
        
        # mse_mu1 = F.mse_loss(student_mu1, teacher_mu1)
        # mse_mu0 = F.mse_loss(student_mu0, teacher_mu0)
        # mse_uplift = F.mse_loss(student_outputs['uplift'], teacher_outputs['uplift'])
        
        response_distill_loss = (kl_mu1 + kl_mu0 + kl_prpsy) / 3
        
        return response_distill_loss
        

    def _compute_feature_distill_loss(self, student_outputs, teacher_outputs):
            """
            feature-based дистилляции (FitNet).
            """
            if 'intermediates' not in student_outputs or 'intermediates' not in teacher_outputs:
                print("Warning: intermediates not found in outputs. Feature distillation will be skipped.")
                return torch.tensor(0.0).to(self.device)
    
            student_share_intermediates = student_outputs['intermediates']['share']
            teacher_share_intermediates = teacher_outputs['intermediates']['share']
            
            feature_distill_loss = 0.0
            num_matches = 0
            
            layer_mapping = {0: 0, 1: 2, 2: 5}  # student_idx: teacher_idx
            
            for student_idx, teacher_idx in layer_mapping.items():
                student_key = f'layer_{student_idx}'
                teacher_key = f'layer_{teacher_idx}'
                
                if student_key in student_share_intermediates and teacher_key in teacher_share_intermediates:
                    student_layer = student_share_intermediates[student_key]
                    teacher_layer = teacher_share_intermediates[teacher_key]
                    
                    if student_key in self.layer_adapters:
                        student_layer = self.layer_adapters[student_key](student_layer)
    
                    mse = F.mse_loss(student_layer, teacher_layer)
                    feature_distill_loss += mse
                    num_matches += 1
    
            if num_matches > 0:
                feature_distill_loss /= num_matches
            
            return feature_distill_loss
