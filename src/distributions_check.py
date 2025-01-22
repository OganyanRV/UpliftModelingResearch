from scipy.stats import ks_2samp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


def check_feature_distributions_by_stat_test(train_data, test_data, print_ =False, plot=False):
    """
    Проверяет различия в распределении фичей и таргетов между тренировочным и тестовым наборами.
    """
    train_data = train_data.values
    test_data = test_data.values

    if train_data.shape[1] != test_data.shape[1]:
        raise ValueError("Тренировочный и тестовый датасеты имеют разное количество фичей!")

    n_features = train_data.shape[1]
    significant_differences = []

    # Проверяем каждую фичу
    for i in range(n_features):
        train_feature = train_data[:, i]
        test_feature = test_data[:, i]

        # Тест Колмогорова-Смирнова для сравнения распределений
        statistic, p_value = ks_2samp(train_feature, test_feature)
        if print_ == True:
            print(f"Фича {i + 1}: KS-статистика={statistic:.4f}, p-value={p_value:.4e}")
        
        # Если p-value очень маленькое, распределения различные
        if p_value < 0.05:
            significant_differences.append(i)

        if plot == True:
            plt.figure(figsize=(10, 5))
            plt.hist(train_feature, bins=30, alpha=0.5, label="Train", color='blue', density=True)
            plt.hist(test_feature, bins=30, alpha=0.5, label="Test", color='orange', density=True)
            plt.title(f"Распределение для фичи {i + 1}")
            plt.legend()
            plt.show()

    if significant_differences:
        print("\nОбнаружены значимые различия в распределениях для следующих фичей:")
        print(", ".join([f"Фича {i + 1}" for i in significant_differences]))
    else:
        print("\nРаспределения похожи для всех фичей (p-value >= 0.05).")



def check_feature_distributions_by_model(train_data, test_data):
    """
    Проверяет различия между тренировочным и тестовым датасетами с помощью CatBoost.
    """

    if train_data.shape[1] != test_data.shape[1]:
        raise ValueError("Количество фичей в тренировочном и тестовом наборах не совпадает!")

    train_data = train_data.copy(deep=True)
    test_data = test_data.copy(deep=True)
    train_data['target'] = 0
    test_data['target'] = 1

    full_data = pd.concat([train_data, test_data], ignore_index=True)
    full_data = full_data if len(full_data) <= 1000000 else full_data.sample(n=1000000)
    X = full_data.drop(columns=['target']).values
    y = full_data['target'].values

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


    model = CatBoostClassifier(verbose=0, iterations=30, random_seed=42)
    model.fit(X_train, y_train)

    # Оцениваем качество классификации с использованием AUC
    y_val_pred = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_val_pred)

    print(f"AUC: {auc:.4f}")
    if 0.45 < auc < 0.55:  # Чуть выше 0.5 допускается из-за случайной ошибки
        print("Тренировочные и тестовые датасеты ПОХОЖИ!")
    else:
        print("Тренировочные и тестовые датасеты РАЗЛИЧАЮТСЯ!")
