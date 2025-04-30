import os
import json
import time
import torch
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Union, Optional, Tuple
from torch.utils.data import DataLoader, TensorDataset as TorchDataset
from src.models.IModelUplift import IModelUplift

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
        
        if from_load and path:
            self.load(path)
        elif config_json:
            if isinstance(config_json, str):
                self.config = json.loads(config_json)
            else:
                self.config = config_json
                
            # Установка устройства для вычислений
            self.device = torch.device('cuda' if torch.cuda.is_available() and 
                                       self.config.get('use_gpu', True) else 'cpu')
            
            # Инициализация модели
            self._initialize_model()
            
            # Настройка оптимизатора и планировщика скорости обучения
            self._setup_optimizer_and_scheduler()
        else:
            raise ValueError("Either config_json or from_load with path must be provided")
    
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
    
    def _get_data_loader(self, X: TorchDataset, batch_size=None, shuffle=False):
        """
        Создание дата лоадера.
        """
        if batch_size is None:
            batch_size = self.config.get('batch_size', 32)
            
        return DataLoader(
            X, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            num_workers=self.config.get('num_workers', 0)
        )
    
    
    def _compute_loss(self, outputs, outcome, treatment):
        """
        Вычисление лосса
        """
        pass
    
    
    def fit(self, X: TorchDataset):
        """
        Обучение модели.
        Возвращает историю обучения
        """
        # Перевод модели в режим обучения
        self.model.train()
        
        # Настройка параметров обучения
        epochs = self.config.get('epochs', 100)
        batch_size = self.config.get('batch_size', 32)
        early_stopping_patience = self.config.get('early_stopping_patience', 10)
        
        # Подготовка загрузчика данных
        train_loader = self._prepare_data_loader(X, batch_size, shuffle=True)
        
        # Подготовка для ранней остановки
        best_loss = float('inf')
        patience_counter = 0
        
        # История обучения
        history = {
            'epoch': [],
            'train_loss': [],
            'learning_rate': []
        }
        
        # Основной цикл обучения
        for epoch in range(epochs):
            # Метрики текущей эпохи
            epoch_loss = 0.0
            num_batches = 0
            
            # Обработка батчей
            for batch in train_loader:
                # Извлечение данных
                features, treatment, outcome = batch
                
                # Обнуление градиентов
                self.optimizer.zero_grad()
                
                # Прямой проход
                outputs = self.model(features)
                
                # Вычисление функции потерь
                loss = self._compute_loss(outputs, outcome, treatment)
                
                # Обратное распространение ошибки
                loss.backward()
                
                # Оптимизация весов
                self.optimizer.step()
                
                # Накопление метрик
                epoch_loss += loss.item()
                num_batches += 1
            
            # Средняя потеря по эпохе
            avg_loss = epoch_loss / num_batches
            
            # Обновление планировщика скорости обучения
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(avg_loss)
                else:
                    self.scheduler.step()
            
            # Запись истории обучения
            history['epoch'].append(epoch + 1)
            history['train_loss'].append(avg_loss)
            history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            
            # Вывод информации о текущей эпохе
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, "
                  f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Проверка ранней остановки
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                # Сохранение лучшей модели
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    # Восстановление лучшей модели
                    self.model.load_state_dict(best_model_state)
                    break
        
        return history
        
    def predict(self, X: TorchDataset):
        """
        Предсказание вероятностей и аплифт-скоров.
        
        Args:
            X: TensorDataset с данными для предсказания (features, treatment)
            
        Returns:
            Словарь с предсказанными значениями
        """
        # Перевод модели в режим оценки
        self.model.eval()
        
        # Подготовка загрузчика данных
        batch_size = self.config.get('inference_batch_size', 64)
        data_loader = self._prepare_data_loader(X, batch_size, shuffle=False)
        
        # Списки для сбора результатов
        y0_list, y1_list, uplift_list = [], [], []
        
        # Прогнозирование без вычисления градиентов
        with torch.no_grad():
            for batch in data_loader:
                features, treatment, _ = batch
                
                # Получение предсказаний
                outputs = self.model(features)
                
                # Сбор результатов
                y0_list.append(outputs['y0'].cpu())
                y1_list.append(outputs['y1'].cpu())
                uplift_list.append(outputs['uplift'].cpu())
        
        # Объединение результатов
        y0 = torch.cat(y0_list, dim=0).numpy()
        y1 = torch.cat(y1_list, dim=0).numpy()
        uplift = torch.cat(uplift_list, dim=0).numpy()
        
        return {
            'y0': y0,
            'y1': y1,
            'uplift': uplift
        }
    
    def predict_light(self, X: TorchDataset):
        """
        Легкая версия предсказания (без возврата значений).
        
        Args:
            X: TensorDataset с данными для предсказания
        """
        # Перевод модели в режим оценки
        self.model.eval()
        
        # Подготовка загрузчика данных
        batch_size = self.config.get('inference_batch_size', 64)
        data_loader = self._prepare_data_loader(X, batch_size, shuffle=False)
        
        # Прогнозирование без вычисления градиентов и сохранения результатов
        with torch.no_grad():
            for batch in data_loader:
                features, treatment, _ = batch
                _ = self.model(features)
    
    def save(self, path):
        """
        Сохранение модели в файл.
        
        Args:
            path: путь для сохранения модели
        """
        # Создание директории, если она не существует
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Подготовка данных для сохранения
        save_data = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }
        
        if self.scheduler is not None:
            save_data['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Сохранение в файл
        torch.save(save_data, path)
        print(f"Model saved to {path}")
    
    def load(self, path):
        """
        Загрузка модели из файла.
        
        Args:
            path: путь к файлу модели
        """
        # Проверка существования файла
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        
        # Загрузка данных
        checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
        
        # Восстановление конфигурации
        self.config = checkpoint['config']
        
        # Установка устройства
        self.device = torch.device('cuda' if torch.cuda.is_available() and 
                                  self.config.get('use_gpu', True) else 'cpu')
        
        # Инициализация модели
        self._initialize_model()
        
        # Загрузка весов модели
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Настройка оптимизатора и загрузка состояния
        self._setup_optimizer_and_scheduler()
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Загрузка состояния планировщика (если он есть)
        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"Model loaded from {path}")
    
    def measure_inference_time(self, data, batch_size, max_size=None):
        """
        Измерение среднего времени инференса модели на данных.
        
        Args:
            data: TensorDataset с данными для измерения
            batch_size: размер батча
            max_size: максимальное количество примеров для измерения
            
        Returns:
            Словарь с результатами измерения (время в секундах на примеры)
        """
        if max_size is not None and len(data) > max_size:
            # Создаем подвыборку данных для измерения
            indices = torch.randperm(len(data))[:max_size]
            subset_data = torch.utils.data.Subset(data, indices)
            data_loader = self._prepare_data_loader(subset_data, batch_size, shuffle=False)
        else:
            data_loader = self._prepare_data_loader(data, batch_size, shuffle=False)
        
        # Прогрев модели
        self.predict_light(torch.utils.data.Subset(data, range(min(10, len(data)))))
        
        # Перевод модели в режим оценки
        self.model.eval()
        
        # Измерение времени
        start_time = time.time()
        self.predict_light(data_loader.dataset)
        end_time = time.time()
        
        # Расчет метрик
        total_time = end_time - start_time
        num_examples = len(data_loader.dataset)
        time_per_example = total_time / num_examples
        examples_per_second = num_examples / total_time
        
        return {
            'total_time_seconds': total_time,
            'num_examples': num_examples,
            'time_per_example_seconds': time_per_example,
            'examples_per_second': examples_per_second
        }
    
    @staticmethod
    def generate_config(count, **params):
        """
        Генерация набора конфигураций для различных моделей.
        
        Args:
            count: количество конфигураций для генерации
            **params: дополнительные параметры и диапазоны для конфигураций
            
        Returns:
            Список словарей с конфигурациями
        """
        configs = []
        
        # Базовая конфигурация
        base_config = {
            'batch_size': 64,
            'epochs': 100,
            'early_stopping_patience': 10,
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
            'num_workers': 2,
            'inference_batch_size': 128
        }
        
        # Объединение базовой конфигурации с переданными параметрами
        for key, value in params.items():
            if isinstance(value, list):
                # Если передан список значений, будем перебирать их
                base_config[key] = value[0]  # Используем первое значение как базовое
            else:
                base_config[key] = value
        
        # Генерация вариаций конфигураций
        for i in range(count):
            config = base_config.copy()
            
            # Модификация конфигурации на основе переданных параметров
            for key, value in params.items():
                if isinstance(value, list):
                    # Выбираем случайное значение из списка
                    config[key] = np.random.choice(value)
                elif isinstance(value, tuple) and len(value) == 2:
                    # Если передан диапазон (min, max), генерируем случайное значение
                    min_val, max_val = value
                    if isinstance(min_val, int) and isinstance(max_val, int):
                        config[key] = np.random.randint(min_val, max_val + 1)
                    else:
                        config[key] = np.random.uniform(min_val, max_val)
            
            configs.append(config)
        
        return configs
