import os
import json
import time
import torch
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Union, Optional, Tuple
from torch.utils.data import DataLoader
from src.models.IModelUplift import IModelUplift
from src.datasets import TorchDataset
from src.metric import get_auuc_v2

class INNUpliftModeling(IModelUplift):
    """
    Родительский класс для реализации нейросетевых моделей аплифт-моделирования.
    """
    
    def __init__(self, config_json=None, from_load=False, path=None):
        """
        Инициализация объекта модели.
        
        Args:
            config_json: строка с JSON-конфигурацией модели
            from_load: флаг, указывающий, что модель загружается из файла
            path: путь для загрузки модели
        """
        super().__init__(config_json, from_load, path)

        if from_load == False:
            if config_json is None:
                raise ValueError(f"No config while contstructing model.")

            if isinstance(config_json, str):
                self.config = json.loads(config_json)
            else:
                self.config = config_json
            self.model = None
            self.device = torch.device('cuda' if torch.cuda.is_available() and self.config.get('use_gpu', True) else 'cpu')
            self._initialize_model()
            self._setup_optimizer_and_scheduler()
            self.history = {}
        else:
            if path is None:
                raise ValueError(f"No config or model paths while contstructing model.")
            self.load(path)
    
    def _initialize_model(self):
        pass
    
    def _setup_optimizer_and_scheduler(self):
        """
        Инициализация оптимизатора и планировщика лр.
        """
        optimizer_config = self.config.get('optimizer', {})
        optimizer_name = optimizer_config.get('name', 'Adam')
        lr = optimizer_config.get('lr', 0.001)
        weight_decay = optimizer_config.get('weight_decay', 0.0)
        
        if optimizer_name == 'Adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), 
                lr=lr, 
                weight_decay=weight_decay
            )
        elif optimizer_name == 'SGD':
            momentum = optimizer_config.get('momentum', 0.9)
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), 
                lr=lr, 
                momentum=momentum, 
                weight_decay=weight_decay
            )
        elif optimizer_name == 'AdamW':
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), 
                lr=lr, 
                weight_decay=weight_decay
            )
        else:
                self.optimizer = torch.optim.AdamW(
                self.model.parameters(), 
                lr=lr, 
                weight_decay=weight_decay
            )
        
        scheduler_config = self.config.get('scheduler', {})
        scheduler_name = scheduler_config.get('name')
        
        if scheduler_name == 'ReduceLROnPlateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=scheduler_config.get('mode', 'min'),
                factor=scheduler_config.get('factor', 0.1),
                patience=scheduler_config.get('patience', 10),
                verbose=scheduler_config.get('verbose', True)
            )
        elif scheduler_name == 'CosineAnnealingLR':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=scheduler_config.get('T_max', 100),
                eta_min=scheduler_config.get('eta_min', 0)
            )
        elif scheduler_name is None:
            self.scheduler = None
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_name}")
    
    def _prepare_data_loader(self, X: TorchDataset, batch_size=None, shuffle=False):
        """
        Подготовка DataLoader
        """
        if batch_size is None:
            batch_size = self.config.get('batch_size', 32)
            
        return DataLoader(
            X, 
            batch_size=batch_size, 
            shuffle=shuffle,
            drop_last=True,
            num_workers=self.config.get('num_workers', 0)
        )
    
    
    def _compute_loss(self, outputs, outcome, treatment):
        """
        Вычисление лосса
        """
        pass

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
                
                outputs = self.model(features)
                
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
            
                outputs = self.model(features)                
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
        
        y0_list, y1_list, uplift_list, treatment_list, outcome_list = [], [], [], [], []
        
        with torch.no_grad():
            for batch in data_loader:
                features, treatment, outcome = batch
                
                outputs = self.model(features)
                
                y0_list.append(outputs['y0'].cpu())
                y1_list.append(outputs['y1'].cpu())
                uplift_list.append(outputs['uplift'].cpu())
                treatment_list.append(treatment.cpu())
                outcome_list.append(outcome.cpu())
        y0 = torch.cat(y0_list, dim=0).numpy()
        y1 = torch.cat(y1_list, dim=0).numpy()
        uplift = torch.cat(uplift_list, dim=0).numpy()
        treatment = torch.cat(treatment_list, dim=0).numpy()
        outcome = torch.cat(outcome_list, dim=0).numpy()

        df = pd.DataFrame({'score': uplift[:, 0], 'treatment': treatment, 'target':outcome})
        return {
            'y0': y0,
            'y1': y1,
            'uplift': uplift,
            'df': df
        }
    
    def predict_light(self, X: DataLoader):
        pass
    #     """
    #     Легкая версия предсказания (без возврата значений).
    #     """
    #     self.model.eval()
        
    #     with torch.no_grad():
    #         for batch in X:
    #             features, treatment, _ = batch
    #             _ = self.model(features)
    
    def save(self, path_current_setup):
        """
        Сохранение модели в файл.
        """
        path = os.path.join(path_current_setup, "model.pkl")
        
        # Подготовка данных для сохранения
        save_data = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'history': self.history
        }
        
        if self.scheduler is not None:
            save_data['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(save_data, path)
    
    def load(self, path_current_setup):
        """
        Загрузка модели из файла.
        """
        path = os.path.join(path_current_setup, "model.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
            
        checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
        self.config = checkpoint['config']
        
        self.device = torch.device('cuda' if torch.cuda.is_available() and 
                                  self.config.get('use_gpu', True) else 'cpu')
        
        self._initialize_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self._setup_optimizer_and_scheduler()
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.history = checkpoint['history']
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
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
            features, _, _ = batch
            start_time = time.time()
            _ = self.model(features)
            end_time = time.time() 
            
            inference_times.append((end_time - start_time) * 1000 / batch_size)
    
            cur_size += batch_size
            if cur_size >= max_size:
                break
    
        mean_inference_time = np.mean(inference_times)
        return mean_inference_time
    
    @staticmethod
    def generate_config(count, params):
        """
        Генерация набора конфигураций для различных моделей.
        count: количество конфигураций для генерации
        **params: дополнительные параметры и диапазоны для конфигураций
        """
        configs = []
        
        base_config = {
            'batch_size': 64,
            'epochs': 10,
            'early_stopping_patience': 2,
            'optimizer': {
                'name': 'Adam',
                'lr': 0.001,
                'weight_decay': 0.0001
            },
            'scheduler': {
                'name': 'ReduceLROnPlateau',
                'patience': 5,
                'factor': 0.5
            },
            'use_gpu': True,
            'num_workers': 0,
            'inference_batch_size': 32
        }
        
        for key, value in params.items():
            if isinstance(value, list):
                base_config[key] = value[0]
            else:
                base_config[key] = value
        
        # Генерация вариаций конфигураций
        for i in range(count):
            config = base_config.copy()
            
            for key, value in params.items():
                if isinstance(value, list):
                    config[key] = np.random.choice(value)
                elif isinstance(value, tuple) and len(value) == 2:
                    min_val, max_val = value
                    if isinstance(min_val, int) and isinstance(max_val, int):
                        config[key] = np.random.randint(min_val, max_val + 1)
                    else:
                        config[key] = np.random.uniform(min_val, max_val)
            
            configs.append(config)
        
        return configs
