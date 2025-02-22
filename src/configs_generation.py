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


def generate_random_configs_tmodel(treatment_parameters, control_parameters, count):
    configs = []
    for _ in range(count):
        treatment_config = generate_random_config_catboost(treatment_parameters)
        control_config = generate_random_config_catboost(control_parameters)

        config = {
                    "lvl_0": {
                        "meta": {
                            "control_name": 0
                        }
                    },
                    "lvl_1": {
                        "treatment": treatment_config,
                        "control": control_config
                    }
                }

        configs.append(config)
    return configs