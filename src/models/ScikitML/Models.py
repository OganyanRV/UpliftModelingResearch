from src.models.ScikitML.IScikitML import IScikitML
from src.configs_generation import generate_random_config_xgboost, generate_random_config_catboost, generate_random_config_catboost_reg
from src.datasets import NumpyDataset

from catboost import CatBoostClassifier, CatBoostRegressor
from xgboost import XGBClassifier, XGBRegressor

from catboost import CatBoostClassifier, CatBoostRegressor
from xgboost import XGBClassifier
from sklift.models import ClassTransformation, ClassTransformationReg
from src.datasets import NumpyDataset
import numpy as np
import os
import pickle
import json

class ClassTransformationModel(IScikitML):
    """
    ClassTransformation-моделинг с помощью scikit-learn.
    """

    def __init__(self, config_json=None, from_load=False, path=None):
        super().__init__(config_json, from_load, path)

        if from_load==False:
            self.model = ClassTransformation(
                estimator=CatBoostClassifier(verbose=0, **self.config['lvl_0']['meta'])
            )

    @staticmethod
    def generate_config(count, **params):
        configs = []
        for _ in range(count):
            config = generate_random_config_catboost(params)
    
            config = {
                        "lvl_0": {
                            "meta": config,
                        }
                    }
    
            configs.append(config)
        return configs

class ClassTransformationRegModel(IScikitML):
    """
    ClassTransformation-моделинг для случая propensity != 0.5 с помощью scikit-learn.
    """

    def __init__(self, config_json=None, from_load=False, path=None):
        super().__init__(config_json, from_load, path)

        if from_load==False:
            self.model = ClassTransformationReg(
                estimator=CatBoostRegressor(verbose=0, **self.config['lvl_0']['meta']),
                propensity_estimator = CatBoostClassifier(iterations=50, verbose=False)
            )

    @staticmethod
    def generate_config(count, **params):
        configs = []
        for _ in range(count):
            config = generate_random_config_catboost_reg(params)
    
            config = {
                        "lvl_0": {
                            "meta": config,
                        }
                    }
    
            configs.append(config)
        return configs
