from src.models.ICausalML.ICausalML import ICausalML
from src.models.ICausalML.ICausalMLPropensity import ICausalMLPropensity
from catboost import CatBoostClassifier, CatBoostRegressor
import causalml
import causalml.metrics as cmetrics
import causalml.inference.tree as ctree
import causalml.inference.meta.tlearner as tlearner
import causalml.inference.meta.slearner as slearner
import causalml.inference.meta.rlearner as rlearner
import causalml.inference.meta.xlearner as xlearner
from causalml.inference.tree import UpliftTreeClassifier, UpliftRandomForestClassifier
from src.datasets import NumpyDataset
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
                learner=CatBoostClassifier(verbose=0, **self.config['lvl_1']['meta']),
                **self.config['lvl_0']['meta']
            )

class SModel(ICausalML):
    """
    s-моделинг с помощью causalml.
    """

    def __init__(self, config_json=None, from_load=False, path=None):
        super().__init__(config_json, from_load, path)

        if from_load==False:
            self.model = slearner.BaseSClassifier(
                learner=CatBoostClassifier(verbose=0, **self.config['lvl_1']['meta']),
                allow_writing_files=True,
                **self.config['lvl_0']['meta']
            )

# class RModel(ICausalML):
#     """
#     r-моделинг с помощью causalml.
#     """

#     def __init__(self, config_json=None, from_load=False, path=None):
#         super().__init__(config_json, from_load, path)

#         if from_load==False:
#             self.model = slearner.BaseRClassifier(
#                 control_learner=CatBoostClassifier(verbose=0, **self.config['lvl_1']['control']),
#                 treatment_learner=CatBoostClassifier(verbose=0, **self.config['lvl_1']['treatment']),
#                 **self.config['lvl_0']['meta']
#             )


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