import random

def generate_random_config_catboost(params):
    iterations = random.randint(*params['iterations'])
    learning_rate = round(random.uniform(*params['learning_rate']), 3)
    depth = random.randint(*params['depth'])
    
    config = {
                "iterations": iterations,
                "learning_rate": learning_rate,
                "depth": depth,
                "loss_function": "Logloss",
                "eval_metric": "AUC"
            }

    return config

def generate_random_config_xgboost(params):
    iterations = random.randint(*params['iterations'])
    learning_rate = round(random.uniform(*params['learning_rate']), 3)
    depth = random.randint(*params['depth'])

    config = {
        'n_estimators': iterations,
        'learning_rate': learning_rate,
        'max_depth': depth,
        'objective': 'binary:logistic',
        'eval_metric': 'auc'
    }
    
    return config

def generate_random_config_catboost_reg(params):
    iterations = random.randint(*params['iterations'])
    learning_rate = round(random.uniform(*params['learning_rate']), 3)
    depth = random.randint(*params['depth'])
    
    config = {
                "iterations": iterations,
                "learning_rate": learning_rate,
                "depth": depth
            }

    return config