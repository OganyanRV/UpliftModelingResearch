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
from src.models.NNUpliftModeling.EFIN import *
from src.metric import get_auuc_v2


class EFINUpliftModel(INNUpliftModeling):
    """
    EFIN (Explicit Feature Interaction Network) для аплифт-моделирования.
    """
    
    def _initialize_model(self):
        input_dim = self.config.get('input_dim')
        hc_dim = self.config.get('hc_dim', 128)
        hu_dim = self.config.get('hu_dim', 64)
        act_type = self.config.get('act_type', 'elu')
        version = self.config.get('efin_version', '3')
        
        if input_dim is None:
            raise ValueError("input_dim must be specified in the config")

        self.model = EFIN(
            input_dim=input_dim,
            hc_dim=hc_dim,
            hu_dim=hu_dim,
            act_type=act_type,
            version=version
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
    def generate_config(count, params):
        
        efin_params = {
            'input_dim': 32,  
            'hc_dim': [64, 128, 256],
            'hu_dim': [32, 64, 128], 
            'is_self': [True, False],
            'act_type': ['elu', 'relu', 'sigmoid', 'tanh'], 
            'batch_size': [32, 64, 128],
            'learning_rate': [0.001, 0.01],
            'efin_version': ['3']
        }
        
        for key, value in params.items():
            efin_params[key] = value
        
        return INNUpliftModeling.generate_config(count, efin_params)

    def num_params(self):
        return sum([p.numel() for p in self.model.parameters() if p.requires_grad])


    def fit(self, X_train: TorchDataset):
        """
        Обучение модели с валидацией.            
        История обучения (словарь с метриками по эпохам)
        """
        train_size = int(0.8 * len(X_train))
        val_size = len(X_train) - train_size
        X_train, X_val = torch.utils.data.random_split(X_train, [train_size, val_size])
        
        self.model.train()
        
        epochs = self.config.get('epochs', 10)
        batch_size = self.config.get('batch_size', 32)
        early_stopping_patience = self.config.get('early_stopping_patience', 2)
        accumulation_steps = self.config.get('gradient_accumulation_steps', 1)
        effective_batch_size = batch_size * accumulation_steps
        
        train_loader = self._prepare_data_loader(X_train, batch_size, shuffle=True)
        val_loader = self._prepare_data_loader(X_val, batch_size, shuffle=False)
        
        best_val_loss = float('inf')
        best_val_auuc = float('-inf')
        early_stopping_criterion = self.config.get('early_stopping_criterion', 'loss')  # 'loss' или 'auuc'
        patience_counter = 0
        
        history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'val_auuc': [],
            'learning_rate': []
        }
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            self.model.train()
            
            for batch_idx, batch in enumerate(train_loader):
                self.model.train()
                features, treatment, outcome = batch     
                
                outputs = self.model(features, treatment)
                
                loss = self._compute_loss(outputs, outcome, treatment) / accumulation_steps
                loss.backward()
                
                epoch_loss += loss.item() * accumulation_steps
                num_batches += 1
                
                if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                    if (batch_idx + 1) % max(1, len(train_loader) // 10) == 0:
                        progress = (batch_idx + 1) / len(train_loader) * 100
                        current_loss = epoch_loss / num_batches
                        print(f"Epoch {epoch+1}/{epochs} - {progress:.1f}% - Loss: {current_loss:.4f}")
             
            avg_train_loss = epoch_loss / num_batches

            # ---- validation ----
            print(f"Validation after epoch")
            val_loss, val_auuc = self._evaluate(val_loader)
            
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            
            history['epoch'].append(epoch + 1)
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(val_loss)
            history['val_auuc'].append(val_auuc)
            history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val AUUC: {val_auuc:.4f}, "
                  f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            if early_stopping_criterion == 'loss' and val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                best_model_state = {name: param.clone() for name, param in self.model.state_dict().items()}
            elif early_stopping_criterion == 'auuc' and val_auuc > best_val_auuc:
                best_val_auuc = val_auuc
                patience_counter = 0
                
                best_model_state = {name: param.clone() for name, param in self.model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    
                    self.model.load_state_dict(best_model_state)
                    break
        
        self.history = history

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


    def measure_inference_time(self, data, batch_size, max_size=None):
        """
        Измерение среднего времени инференса модели на данных.
        """
        max_size = 5000
        batch_size=32
        indices = torch.randperm(len(data))[:max_size]
        subset_data = torch.utils.data.Subset(data, indices)
        data_loader = self._prepare_data_loader(subset_data, batch_size, shuffle=False)
        
        self.model.eval()
        
        # Измерение времени
        inference_times = []
    
        cur_size = 0
        for batch in data_loader:
            features, treatment, _ = batch
            start_time = time.time()
            _ = self.model(features, treatment)
            end_time = time.time() 
            
            inference_times.append((end_time - start_time) * 1000 / batch_size)
    
            cur_size += batch_size
            if cur_size >= max_size:
                break
    
        mean_inference_time = np.mean(inference_times)
        return mean_inference_time