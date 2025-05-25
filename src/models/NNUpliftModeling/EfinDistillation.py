import os
import json
import time
import torch
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Union, Optional, Tuple
from torch.utils.data import DataLoader
from src.datasets import TorchDataset, PairedUpliftDataset
import torch.nn as nn
import torch.nn.functional as F
from src.models.NNUpliftModeling.INNUpliftModeling import INNUpliftModeling
from src.metric import get_auuc_v2

class EFIN(nn.Module):
    """
    EFIN (Explicit Feature Interaction Network) для аплифт-моделирования.
    """
    def __init__(self, input_dim, hc_dim, hu_dim, act_type='elu', version='3', return_intermediates=False):
        """
        Инициализация модели EFIN.
        
        Args:
            input_dim: Количество входных признаков
            hc_dim: Размерность скрытых слоев control net
            hu_dim: Размерность скрытых слоев uplift net
            act_type: Тип функции активации
            version: Версия модели ('3' - компактная, '6' - расширенная)
            return_intermediates: Возвращать ли промежуточные активации
        """
        super(EFIN, self).__init__()
        self.nums_feature = input_dim
        self.return_intermediates = return_intermediates
        self.version = version
        
        self.att_embed_1 = nn.Linear(hu_dim, hu_dim, bias=False)
        self.att_embed_2 = nn.Linear(hu_dim, hu_dim)
        self.att_embed_3 = nn.Linear(hu_dim, 1, bias=False)

        # self-attention
        self.softmax = nn.Softmax(dim=-1)
        self.Q_w = nn.Linear(in_features=hu_dim, out_features=hu_dim, bias=True)
        self.K_w = nn.Linear(in_features=hu_dim, out_features=hu_dim, bias=True)
        self.V_w = nn.Linear(in_features=hu_dim, out_features=hu_dim, bias=True)

        # Эмбеды признаков
        self.x_rep = nn.Embedding(input_dim, hu_dim)

        # Эмбеды тритмента
        self.t_rep = nn.Linear(1, hu_dim)

        # Control Net (для предсказания без воздействия)
        if version == '3':
            self.c_layers = nn.ModuleList([
                nn.Linear(input_dim * hu_dim, hc_dim),
                nn.Linear(hc_dim, hc_dim),
                nn.Linear(hc_dim, hc_dim // 2),
                nn.Linear(hc_dim // 2, hc_dim // 4)
            ])
            out_dim = hc_dim // 4
        else: 
            self.c_layers = nn.ModuleList([
                nn.Linear(input_dim * hu_dim, hc_dim),
                nn.Linear(hc_dim, hc_dim),
                nn.Linear(hc_dim, hc_dim // 2),
                nn.Linear(hc_dim // 2, hc_dim // 2),
                nn.Linear(hc_dim // 2, hc_dim // 4),
                nn.Linear(hc_dim // 4, hc_dim // 4)
            ])
            out_dim = hc_dim // 4

        self.c_logit = nn.Linear(out_dim, 1)
        self.c_tau = nn.Linear(out_dim, 1)

        # Uplift Net (для моделирования инкрементального эффекта)
        if version == '3':
            self.u_layers = nn.ModuleList([
                nn.Linear(hu_dim, hu_dim),
                nn.Linear(hu_dim, hu_dim // 2),
                nn.Linear(hu_dim // 2, hu_dim // 4)
            ])
            out_dim = hu_dim // 4
        else:
            self.u_layers = nn.ModuleList([
                nn.Linear(hu_dim, hu_dim),
                nn.Linear(hu_dim, hu_dim // 2),
                nn.Linear(hu_dim // 2, hu_dim // 2),
                nn.Linear(hu_dim // 2, hu_dim // 4),
                nn.Linear(hu_dim // 4, hu_dim // 4)
            ])
            out_dim = hu_dim // 4
        
        self.t_logit = nn.Linear(out_dim, 1)
        self.u_tau = nn.Linear(out_dim, 1)

        if act_type == 'sigmoid':
            self.act = nn.Sigmoid()
        elif act_type == 'tanh':
            self.act = nn.Tanh()
        elif act_type == 'relu':
            self.act = nn.ReLU()
        elif act_type == 'elu':
            self.act = nn.ELU()
        else:
            raise RuntimeError(f'Неизвестный тип активации: {act_type}')

    def self_attn(self, q, k, v):
        Q, K, V = self.Q_w(q), self.K_w(k), self.V_w(v)
        attn_weights = Q.matmul(torch.transpose(K, 1, 2)) / (K.shape[-1] ** 0.5)
        attn_weights = self.softmax(torch.sigmoid(attn_weights))

        outputs = attn_weights.matmul(V)

        return outputs, attn_weights

    def interaction_attn(self, t, x):
        attention = []
        for i in range(self.nums_feature):
            temp = self.att_embed_3(torch.relu(
                torch.sigmoid(self.att_embed_1(t)) + torch.sigmoid(self.att_embed_2(x[:, i, :]))))
            attention.append(temp)
        attention = torch.squeeze(torch.stack(attention, 1), 2)
        attention = torch.softmax(attention, 1)

        outputs = torch.squeeze(torch.matmul(torch.unsqueeze(attention, 1), x), 1)
        return outputs, attention

    def forward(self, features, is_treat):
        intermediates = {'c_layers': {}, 'u_layers': {}} if self.return_intermediates else None
        
        t_true = torch.unsqueeze(is_treat, 1)
        
        x_rep = features.unsqueeze(2) * self.x_rep.weight.unsqueeze(0)

        # Control Net
        dims = x_rep.size()
        _x_rep = x_rep / torch.linalg.norm(x_rep, dim=1, keepdim=True)
        xx, xx_weight = self.self_attn(_x_rep, _x_rep, _x_rep)

        _x_rep = torch.reshape(xx, (dims[0], dims[1] * dims[2]))

        c_last = _x_rep
        for i, layer in enumerate(self.c_layers):
            c_last = layer(c_last)
            if self.return_intermediates:
                intermediates['c_layers'][f'layer_{i}'] = c_last.clone()
            c_last = self.act(c_last)
        
        c_logit = self.c_logit(c_last)
        c_tau = self.c_tau(c_last)
        c_prob = torch.sigmoid(c_logit)

        # Uplift Net
        t_rep = self.t_rep(torch.ones_like(t_true))

        xt, xt_weight = self.interaction_attn(t_rep, x_rep)
        
        u_last = xt
        for i, layer in enumerate(self.u_layers):
            u_last = layer(u_last)
            if self.return_intermediates:
                intermediates['u_layers'][f'layer_{i}'] = u_last.clone()
            u_last = self.act(u_last)

        t_logit = self.t_logit(u_last)
        u_tau = self.u_tau(u_last)
        t_prob = torch.sigmoid(t_logit)
        
        # For predictions
        c_logit_fix = c_logit.detach()
        uc = c_logit
        ut = (c_logit_fix + u_tau)

        outputs = {
            'c_logit': c_logit,   # Логиты для контрольной группы
            'c_prob': c_prob,     # Вероятности для контрольной группы
            'c_tau': c_tau,       # Tau для контрольной группы
            't_logit': t_logit,   # Логиты для группы воздействия
            't_prob': t_prob,     # Вероятности для группы воздействия
            'u_tau': u_tau,       # Tau для uplift
            'uc': uc,             # Logits для y0
            'ut': ut,             # Logits для y1
            'uplift': t_prob - c_prob,  # Предсказание аплифта (p1 - p0)
            'p_mu0': c_prob,       # Вероятность положительного исхода без воздействия
            'p_mu1': t_prob,       # Вероятность положительного исхода с воздействием
            'mu0_logit': c_logit,  # Логиты для исхода без воздействия
            'mu1_logit': t_logit   # Логиты для исхода с воздействием
        }

        if self.return_intermediates:
            outputs['intermediates'] = intermediates
        
        return outputs
        
class EfinDistillation(EFINUpliftModel):
    """
    EFIN модель с возможностью дистилляции знаний от учителя к ученику.
    Поддерживает response-based и feature-based дистилляцию.
    """
    
    def __init__(self, config_json=None, from_load=False, path=None, teacher_model=None):
        """
        Инициализация модели дистилляции EFIN.
        
        Args:
            config_json: строка с JSON-конфигурацией модели ученика
            from_load: флаг, указывающий, что модель загружается из файла
            path: путь для загрузки модели
            teacher_model: учительская модель для дистилляции (EFINUpliftModel)
        """
        super().__init__(config_json, from_load, path)
        
        self.teacher_model = teacher_model
        
  
        self.distill_config = self.config.get('distillation', {})
        self.use_response_distill = self.distill_config.get('use_response_distill', True)
        self.use_feature_distill = self.distill_config.get('use_feature_distill', False)
        self.response_weight = self.distill_config.get('response_weight', 0.5)
        self.feature_weight = self.distill_config.get('feature_weight', 0.5)
        self.temperature = self.distill_config.get('temperature', 3.0)
        

        self._initialize_model()
        self._init_feature_adapters()
        self.teacher_model.model.eval()
    
    def _initialize_model(self):
        """
        Инициализация модели ученика
        """
        input_dim = self.config.get('input_dim')
        hc_dim = self.config.get('hc_dim', 128)
        hu_dim = self.config.get('hu_dim', 64)
        act_type = self.config.get('act_type', 'elu')
        version = '3'

       self.model = EFIN(
            input_dim=input_dim,
            hc_dim=hc_dim,
            hu_dim=hu_dim,
            act_type=act_type,
            version=version,
            return_intermediates=self.use_feature_distill
        ).to(self.device)
    
    def _init_feature_adapters(self):
        """
        Инициализация адаптеров для feature-based дистилляции.
        """
        # В EFIN версии 3 и 6 отличаются количеством слоев в c_layers и u_layers
        # Нам нужно создать адаптеры для промежуточных выходов
        
        self.c_layer_adapters = nn.ModuleDict()
        self.u_layer_adapters = nn.ModuleDict()
        
        # Для control сети (c_layers)
        # В EFIN версии 3: 4 линейных слоя
        # В EFIN версии 6: 6 линейных слоев
        # Соответствие слоев: ученик -> учитель
        # layer_0 -> layer_0
        # layer_1 -> layer_2
        # layer_2 -> layer_4
        # layer_3 -> layer_5
        
        student_hc_dim = self.config.get('hc_dim', 128)
        teacher_hc_dim = self.teacher_model.config.get('hc_dim', 128)
        
        student_hu_dim = self.config.get('hu_dim', 64)
        teacher_hu_dim = self.teacher_model.config.get('hu_dim', 64)
        
        # Карта соответствия слоев ученика к слоям учителя
        c_layer_mapping = {0: 0, 1: 2, 2: 4, 3: 5}
        u_layer_mapping = {0: 0, 1: 2, 2: 4}
        

        for student_idx, teacher_idx in c_layer_mapping.items():
            if student_idx == 0:
                student_dim = student_hc_dim
                teacher_dim = teacher_hc_dim
            elif student_idx == 1:
                student_dim = student_hc_dim
                teacher_dim = teacher_hc_dim
            elif student_idx == 2:
                student_dim = student_hc_dim // 2
                teacher_dim = teacher_hc_dim // 2
            elif student_idx == 3:
                student_dim = student_hc_dim // 4
                teacher_dim = teacher_hc_dim // 4
            
            if student_dim != teacher_dim:
                self.c_layer_adapters[f'layer_{student_idx}'] = nn.Linear(student_dim, teacher_dim).to(self.device)
            else:
                self.c_layer_adapters[f'layer_{student_idx}'] = nn.Identity().to(self.device)

        for student_idx, teacher_idx in u_layer_mapping.items():
            if student_idx == 0:
                student_dim = student_hu_dim
                teacher_dim = teacher_hu_dim
            elif student_idx == 1:
                student_dim = student_hu_dim // 2
                teacher_dim = teacher_hu_dim // 2
            elif student_idx == 2:
                student_dim = student_hu_dim // 4
                teacher_dim = teacher_hu_dim // 4
            
            if student_dim != teacher_dim:
                self.u_layer_adapters[f'layer_{student_idx}'] = nn.Linear(student_dim, teacher_dim).to(self.device)
            else:
                self.u_layer_adapters[f'layer_{student_idx}'] = nn.Identity().to(self.device)
    
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
                    teacher_outputs = self.teacher_model.model(features, treatment)
                
                self.optimizer.zero_grad()
                student_outputs = self.model(features)
                
                task_loss = self._compute_loss(student_outputs, outcome, treatment)
                
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
        
    def _compute_response_distill_loss(self, student_outputs, teacher_outputs):
        student_c_logit = student_outputs['c_logit']
        student_ut = student_outputs['ut']
        student_uc = student_outputs['uc']
        student_uplift = student_outputs['uplift']
        
        teacher_c_logit = teacher_outputs['c_logit']
        teacher_ut = teacher_outputs['ut']
        teacher_uc = teacher_outputs['uc']
        teacher_uplift = teacher_outputs['uplift']
        
        temp = self.temperature
        
        kl_c_logit = F.kl_div(
            F.log_softmax(student_c_logit / temp, dim=1),
            F.softmax(teacher_c_logit / temp, dim=1),
            reduction='batchmean'
        ) * (temp ** 2)
        
        mse_uplift = F.mse_loss(student_uplift, teacher_uplift)
        
        mse_uc = F.mse_loss(student_uc, teacher_uc)
        mse_ut = F.mse_loss(student_ut, teacher_ut)
        
        response_distill_loss = (kl_c_logit + mse_uplift + mse_uc + mse_ut) / 4
        
        return response_distill_loss
    
    def _compute_feature_distill_loss(self, student_outputs, teacher_outputs):
        """
        Feature-based дистилляция (FitNet).
        """
        if 'intermediates' not in student_outputs or 'intermediates' not in teacher_outputs:
            return torch.tensor(0.0).to(self.device)
        
        student_intermediates = student_outputs['intermediates']
        teacher_intermediates = teacher_outputs['intermediates']
        
        feature_distill_loss = 0.0
        count = 0
        
        # Карты соответствия слоев
        c_layer_mapping = {0: 0, 1: 2, 2: 4, 3: 5}  # student -> teacher
        u_layer_mapping = {0: 0, 1: 2, 2: 4}  # student -> teacher
        
        for student_idx, teacher_idx in c_layer_mapping.items():
            student_key = f'layer_{student_idx}'
            teacher_key = f'layer_{teacher_idx}'
            
            if (student_key in student_intermediates['c_layers'] and 
                teacher_key in teacher_intermediates['c_layers']):
                
                student_act = student_intermediates['c_layers'][student_key]
                teacher_act = teacher_intermediates['c_layers'][teacher_key]

                if student_key in self.c_layer_adapters:
                    student_act = self.c_layer_adapters[student_key](student_act)
                
                loss = F.mse_loss(student_act, teacher_act)
                feature_distill_loss += loss
                count += 1
        
        for student_idx, teacher_idx in u_layer_mapping.items():
            student_key = f'layer_{student_idx}'
            teacher_key = f'layer_{teacher_idx}'
            
            if (student_key in student_intermediates['u_layers'] and 
                teacher_key in teacher_intermediates['u_layers']):
                
                student_act = student_intermediates['u_layers'][student_key]
                teacher_act = teacher_intermediates['u_layers'][teacher_key]
                
                if student_key in self.u_layer_adapters:
                    student_act = self.u_layer_adapters[student_key](student_act)
                
                loss = F.mse_loss(student_act, teacher_act)
                feature_distill_loss += loss
                count += 1
        
        if count > 0:
            feature_distill_loss = feature_distill_loss / count
        
        return feature_distill_loss


    def _evaluate(self, data_loader):
        """
        loss и AUUC на валидационном или тестовом наборе.
        """
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        all_uplift_scores = []
        all_treatments = []
        all_outcomes = []
        
        with torch.no_grad():
            for batch in data_loader:
                features, treatment, outcome = batch
            
                outputs = self.model(features, treatment)                
                loss = self._compute_loss(outputs, outcome, treatment)
                total_loss += loss.item()
                num_batches += 1
                
                uplift_scores = outputs["uplift"]
                all_uplift_scores.append(uplift_scores.cpu())
                all_treatments.append(treatment.cpu())
                all_outcomes.append(outcome.cpu())
        
        avg_loss = total_loss / num_batches
        auuc = float('nan')
    
        uplift_scores = torch.cat(all_uplift_scores, dim=0).numpy().flatten()
        treatments = torch.cat(all_treatments, dim=0).numpy().flatten()
        outcomes = torch.cat(all_outcomes, dim=0).numpy().flatten()                
        auuc = get_auuc_v2(uplift_scores, treatments, outcomes)
        
        return avg_loss, auuc
        
    def predict(self, X: TorchDataset):
        """
        Предсказание вероятностей и аплифт-скоров.
        """
        self.model.eval()
        batch_size = self.config.get('inference_batch_size', 32)
        data_loader = self._prepare_data_loader(X, batch_size, shuffle=False)
        
        uplift_list, treatment_list, outcome_list = [], [], []
        
        with torch.no_grad():
            for batch in data_loader:
                features, treatment, outcome = batch
                
                outputs = self.model(features, treatment)
                
                uplift_list.append(outputs['uplift'].cpu())
                treatment_list.append(treatment.cpu())
                outcome_list.append(outcome.cpu())
                
        uplift = torch.cat(uplift_list, dim=0).numpy()
        treatment = torch.cat(treatment_list, dim=0).numpy()
        outcome = torch.cat(outcome_list, dim=0).numpy()

        df = pd.DataFrame({'score': uplift[:, 0], 'treatment': treatment, 'target':outcome})
        return {
            'uplift': uplift,
            'df': df
        }

    def fit_kdsm(self, X_train: PairedUpliftDataset, lambda_kd=0.5):
        """
        KDSM Обучение модели с валидацией.            
        История обучения (словарь с метриками по эпохам)
        """
        train_size = int(0.8 * len(X_train))
        
        self.model.train()
        
        epochs = 2
        batch_size = 1
        early_stopping_patience = 2
        accumulation_steps = 64
        effective_batch_size = batch_size * accumulation_steps
        
        train_loader = self._prepare_data_loader(X_train, batch_size, shuffle=True)
        
        best_loss = float('inf')
        early_stopping_criterion = 'loss'
        patience_counter = 0
        
        history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_kdsm_loss = 0.0
            num_batches = 0
            
            self.model.train()
            
            for batch_idx, batch in enumerate(train_loader):
                X_train.shuffle_pairs()
                train_loader = self._prepare_data_loader(X_train, batch_size, shuffle=True)
                
                self.model.train()
                t_features, t_treatment, t_outcome, t_teacher_pred, c_features, c_treatment, c_outcome, c_teacher_pred = batch
                
                t_outputs = self.model(t_features)
                t_pred = t_outputs['p_mu1']  # Вероятность положительного исхода при воздействии
                
                c_outputs = self.model(c_features)
                c_pred = c_outputs['p_mu0']  # Вероятность положительного исхода без воздействия
                
                student_uplift = t_pred - c_pred
                
                teacher_uplift = t_teacher_pred - c_teacher_pred

                t_loss = self._compute_loss(t_outputs, t_outcome, t_treatment)
                c_loss = self._compute_loss(c_outputs, c_outcome, c_treatment)
                
                kd_loss = F.mse_loss(student_uplift.squeeze(), teacher_uplift.squeeze())
                total_loss = t_loss + c_loss + lambda_kd * kd_loss

                normalized_loss = total_loss / accumulation_steps
                normalized_loss.backward()
                
                epoch_loss += normalized_loss.item() * accumulation_steps
                num_batches += 1

                # if (batch_idx % 50 == 0):
                #     print(batch_idx)
                #     print(kd_loss)
                
                if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    if (batch_idx + 1) % 256 == 0:
                        progress = (batch_idx + 1) / len(train_loader) * 100
                        current_loss = epoch_loss / num_batches
                        print(f"Epoch {epoch+1}/{epochs} - {progress:.1f}% - Loss: {current_loss:.4f}")
             
            avg_train_loss = epoch_loss / num_batches

            
            history['epoch'].append(epoch + 1)
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(val_loss)
            history['val_auuc'].append(val_auuc)
            history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val AUUC: {val_auuc:.4f}, "
                  f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            if early_stopping_criterion == 'loss' and avg_train_loss < best_loss:
                best_loss = avg_train_loss
                patience_counter = 0                    
                best_model_state = {name: param.clone() for name, param in self.model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    
                    self.model.load_state_dict(best_model_state)
                    break
        
        self.history_kdsm = history

    