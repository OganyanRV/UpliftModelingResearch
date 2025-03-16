import time
from src.global_params import *
import numpy as np
import os
import json
import pickle
from src.metric import get_auuc, uplift_by_percentile_CUM
import pandas as pd

def get_paths_train_test(ds_name, features_percent):
    """
    Возвращает путь по имени датасета и проценту фичей в нем
    """

    path_to_data_train = f'data/{ds_name}/{features_percent}/train.tsv'
    path_to_data_test = f'data/{ds_name}/{features_percent}/test.tsv'

    path_to_data_train = BASE_PATH + '/' + path_to_data_train
    path_to_data_test = BASE_PATH + '/' + path_to_data_test

    return path_to_data_train, path_to_data_test


def train_test_model(ds_name, features_percent, factory, config, compressions=None, batch_size=32, max_size=100000):
    """
    Обучает модель, предиктит, сохраняет информацию о модели и добавляет статистики в общую таблицу
    """
    train_path, test_path = get_paths_train_test(ds_name=ds_name, features_percent=features_percent)
    model, train, test = factory.create(config, train_path, test_path)
    model.fit(train)
    predicted = model.predict(test)
    write(model, test, predicted, ds_name=ds_name, features_percent=features_percent, compressions=compressions,
          batch_size=batch_size, max_size=max_size)    
    
def make_stats_table(path):
    table = pd.DataFrame(columns=[
        'Model',
        'Path',
        'Dataset',
        'Features Percent',
        'Latency (ms)',
        'Binary Size (MB)',
        "AUUC (test)",
        "Precision@5",
        "Precision@10",
        "Precision@15",
        "Precision@20",
        "Precision@25",
        "Precision@50",
        "Compressions"    
    ])

    table.to_csv(path, sep="\t", index=False)

def append_exp(model, test, predicted, ds_name, features_percent,
               path_current_setup, compressions,
              batch_size=32, max_size=200000):
    
    path_exps_stats = BASE_PATH + "/" + EXPS_PATH + "/stats.tsv"
    
    model_name = model.__class__.__name__
    inference_time_ms = model.measure_inference_time(test, batch_size, max_size=max_size)
    size_model_mb = os.path.getsize(BASE_PATH + "/" + path_current_setup + "/model.pkl") / 1e6
    auuc_model = get_auuc(predicted)
    compressions = compressions or {}
    uplift_precision = uplift_by_percentile_CUM(predicted[COL_TARGET], predicted["score"],
                                                predicted[COL_TREATMENT], strategy='overall', bins=20)
    uplift_5 = uplift_precision.loc['5'].uplift
    uplift_10 = uplift_precision.loc['15'].uplift
    uplift_15 = uplift_precision.loc['20'].uplift
    uplift_20 = uplift_precision.loc['15'].uplift
    uplift_25 = uplift_precision.loc['25'].uplift
    uplift_50 = uplift_precision.loc['50'].uplift

    if os.path.exists(path_exps_stats) == False:
        make_stats_table(path_exps_stats)

    table = pd.read_csv(path_exps_stats, sep='\t')
    table.loc[len(table)] = [
        model_name,
        path_current_setup,
        ds_name,
        features_percent,
        inference_time_ms,
        size_model_mb,
        auuc_model,
        uplift_5,
        uplift_10,
        uplift_15,
        uplift_20,
        uplift_25,
        uplift_50,
        compressions
    ]
    table.to_csv(path_exps_stats, sep='\t', index=False)

    return path_exps_stats
    

def write_files(model, predictions, ds_name, features_percent):
    """
    Функция создает папку в нужной директории и записывает туда бинарик модели, предикты модели и конфиг.
    """

    path_overall_stats = BASE_PATH + "/" + EXPS_PATH
    
    free_folder_number = 0
    os.makedirs(f'{path_overall_stats}/{ds_name}', exist_ok=True)
    os.makedirs(f'{path_overall_stats}/{ds_name}/{features_percent}', exist_ok=True)        
    while os.path.exists(os.path.join(f'{path_overall_stats}/{ds_name}/{features_percent}', str(free_folder_number))):
        free_folder_number += 1
    path_current_setup = f'{path_overall_stats}/{ds_name}/{features_percent}/{free_folder_number}'    
    os.makedirs(path_current_setup, exist_ok=True)

    # Сохранение модели конфига и тп
    model.save(path_current_setup)
    
    # Сохранение предсказаний
    predictions_path = os.path.join(path_current_setup, "predictions.tsv")
    predictions.to_csv(predictions_path, sep='\t', index=False)

    return EXPS_PATH + "/" + f"{ds_name}/{features_percent}/{free_folder_number}"


def write(model, test, predictions, ds_name, features_percent, compressions= None,
         batch_size=32, max_size=100000):
    """
    Сохраняет метрики в общую таблицу и модель с конфигом для каждого отдельного запуска
    model - модель
    predictions - таблица вида score treatment target
    ds_name - название тестируемого датасета
    features_percent - процент фичей в датасете
    compressions - конфиг компрессий
    batch_size, max_size - параметры для тестирования инференса модели
    """
    path_current_setup = write_files(model, predictions, ds_name, features_percent)
    print(f"Модель, предсказания и конфиг сохранены в директории {path_current_setup}")
    path_exp = append_exp(model, test, predictions, ds_name, features_percent, path_current_setup, compressions, batch_size, max_size)
    print(f"Эксперимент сохранен в таблице {path_exp}")