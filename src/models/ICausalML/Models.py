from src.models.ICausalML.ICausalML import ICausalML
from catboost import CatBoostClassifier
import causalml
import causalml.metrics as cmetrics
import causalml.inference.tree as ctree
import causalml.inference.meta.tlearner as tlearner
import causalml.inference.meta.slearner as slearner
import causalml.inference.meta.rlearner as rlearner
import causalml.inference.meta.xlearner as xlearner
from causalml.inference.tree import UpliftTreeClassifier, UpliftRandomForestClassifier
# Конкретная реализация модели
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