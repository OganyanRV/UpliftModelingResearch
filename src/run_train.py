from pathlib import Path
import sys
 
if sys.argv:
    sys.path.insert(0, str(Path('/Users/ogrobertino/UpliftModelingResearch/').resolve()))
    
import argparse
from tqdm import tqdm
from src.utils import get_paths_train_test, train_test_model
from src.factory import SModelFactory
from src.models.CausalML.Models import SModel



def main(args):
    # Упаковка параметров в словарь для генерации конфигураций
    parameters = {
        'iterations': (args.iterations[0], args.iterations[1]),
        'learning_rate': (args.learning_rate[0], args.learning_rate[1]),
        'depth': (args.depth[0], args.depth[1])
    }
    
    # Генерация конфигураций модели
    model_class = getattr(__import__('src.factory', fromlist=[args.model]), args.model)
    factory_class = getattr(__import__('src.factory', fromlist=[args.model + "Factory"]), args.model + "Factory")
    
    configs = model_class.generate_config(count=args.count, **parameters)

    # Обучение и тестирование моделей с заданными параметрами
    for i in tqdm(range(len(configs)), desc="Training models"):
        train_test_model(ds_name=args.ds_name, features_percent=args.features_percent, factory=factory_class, config=configs[i])    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train models with specified parameters.")
    
    # Парсер для параметров
    parser.add_argument('--iterations', type=int, nargs=2, default=(30, 70), help="Range of iterations as two integers.")
    parser.add_argument('--learning_rate', type=float, nargs=2, default=(0.01, 0.5), help="Range of learning rates as two floats.")
    parser.add_argument('--depth', type=int, nargs=2, default=(4, 15), help="Range of depths as two integers.")
    parser.add_argument('--ds_name', type=str, required=True, help="Name of the dataset.")
    parser.add_argument('--features_percent', type=int, default=100, help="Percent of features to use.")
    parser.add_argument('--count', type=int, default=1, help="Count of trainings to run.")
    parser.add_argument('--model', type=str, required=True, help="Model name.")
    
    args = parser.parse_args()
    
    # Вызов основной функции с аргументами
    main(args)