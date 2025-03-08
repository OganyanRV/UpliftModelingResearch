from src.models.CausalML.ICausalML import ICausalML
from src.models.CausalML.ICausalMLPropensity import ICausalMLPropensity
from src.configs_generation import generate_random_config_xgboost, generate_random_config_catboost, generate_random_config_catboost_reg
from src.datasets import NumpyDataset

from catboost import CatBoostClassifier, CatBoostRegressor
from xgboost import XGBClassifier, XGBRegressor
import causalml
import causalml.metrics as cmetrics
import causalml.inference.tree as ctree
import causalml.inference.meta.tlearner as tlearner
import causalml.inference.meta.slearner as slearner
import causalml.inference.meta.rlearner as rlearner
import causalml.inference.meta.xlearner as xlearner
import causalml.inference.meta.drlearner as drlearner
from causalml.inference.tree import UpliftTreeClassifier, UpliftRandomForestClassifier
import numpy as np
import os
import pickle
import json

class TModel(ICausalML):
    """
    t-моделинг с помощью causalml.
    """

    def __init__(self, config_json=None, from_load=False, path=None):
        super().__init__(config_json, from_load, path)

        if from_load==False:
            self.model = tlearner.BaseTClassifier(
                control_learner=CatBoostClassifier(verbose=0, **self.config['lvl_1']['control']),
                treatment_learner=CatBoostClassifier(verbose=0, **self.config['lvl_1']['treatment']),
                **self.config['lvl_0']['meta']
            )
            
    @staticmethod
    def generate_config(count, **params):
        configs = []
        for _ in range(count):
            treatment_config = generate_random_config_catboost(params)
            control_config = generate_random_config_catboost(params)
    
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

class SModel(ICausalML):
    """
    s-моделинг с помощью causalml.
    """

    def __init__(self, config_json=None, from_load=False, path=None):
        super().__init__(config_json, from_load, path)

        if from_load==False:
            self.model = slearner.BaseSClassifier(
                learner=XGBClassifier(verbosity=0, **self.config['lvl_1']['meta']),
                **self.config['lvl_0']['meta']
            )
            
    @staticmethod
    def generate_config(count, **params):
        configs = []
        for _ in range(count):
            config = generate_random_config_xgboost(params)
    
            config = {
                        "lvl_0": {
                            "meta": {
                                "control_name": 0
                            }
                        },
                        "lvl_1": {
                            "meta": config
                        }
                    }
    
            configs.append(config)
        return configs

class XModel(ICausalMLPropensity):
    """
    x-моделинг с помощью causalml.
    """
    def __init__(self, config_json=None, from_load=False, path=None):
        super().__init__(config_json, from_load, path)

        
        if from_load==False:
            self.model = xlearner.BaseXClassifier(
                outcome_learner=CatBoostClassifier(verbose=0, **self.config['lvl_1']['outcome']),
                effect_learner=CatBoostRegressor(verbose=0, **self.config['lvl_1']['effect']),
                **self.config['lvl_0']['meta']
            )

    @staticmethod
    def generate_config(count, **params):
        configs = []
        for _ in range(count):
            config_outcome = generate_random_config_catboost(params)
            config_effect = generate_random_config_catboost_reg(params)
    
            config = {
                        "lvl_0": {
                            "meta": {
                                "control_name": 0
                            }
                        },
                        "lvl_1": {
                            "outcome": config_outcome,
                            "effect": config_effect
                        }
                    }
    
            configs.append(config)
        return configs


class DRModel(ICausalMLPropensity):
    """
    dr-моделинг с помощью causalml.
    """

    def __init__(self, config_json=None, from_load=False, path=None):
        super().__init__(config_json, from_load, path)

        if from_load==False:
            self.model = drlearner.BaseDRLearner(
                learner=CatBoostRegressor(verbose=0, **self.config['lvl_1']['meta']),
                **self.config['lvl_0']['meta']
            )

    @staticmethod
    def generate_config(count, **params):
        configs = []
        for _ in range(count):
            config = generate_random_config_catboost_reg(params)
    
            config = {
                        "lvl_0": {
                            "meta": {
                                "control_name": 0
                            }
                        },
                        "lvl_1": {
                            "meta": config
                        }
                    }
    
            configs.append(config)
        return configs
        