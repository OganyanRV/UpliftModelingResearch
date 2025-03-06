from src.models.ScikitML.IScikitML import IScikitML
from catboost import CatBoostClassifier, CatBoostRegressor
from xgboost import XGBClassifier
from sklift.models import ClassTransformation
from src.datasets import NumpyDataset
import numpy as np
import os
import pickle
import json

class ClassTransformationModel(ScikitML):
    """
    ClassTransformation-моделинг с помощью scikit-learn.
    """

    def __init__(self, config_json=None, from_load=False, path=None):
        super().__init__(config_json, from_load, path)

        if from_load==False:
            self.model = ClassTransformation(
                estimator=XGBClassifier(verbose=0, **self.config['lvl_1']['meta']),
                **self.config['lvl_0']['meta']
            )
