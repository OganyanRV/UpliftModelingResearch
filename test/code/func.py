import pandas as pd
from pathlib import Path

# def f():
#     return pd.read_csv('../data/ds.csv')

def f():
    # Получаем путь к директории, где находится func2.py
    current_dir = Path(__file__).parent
    # Строим абсолютный путь к данным
    data_path = current_dir / "../data/ds.csv"
    return pd.read_csv(data_path.resolve())  # resolve() убирает '../'

def k():
    current_dir = Path(__file__).parent
    data_path = current_dir / "../data/ds.csv"
    
    return Path(__file__).parent, data_path.resolve()

if __name__ == "main":
    print(f())
    print('kek')