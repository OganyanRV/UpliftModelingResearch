from src.models.IModelUplift import IModelUplift
from src.datasets import NumpyDataset
import time
import numpy as np
import os
import pickle
import json

class IScikitML(IModelUplift):
    """
        Родительский класс для реализации классических моделей аплифт-моделирования с помощью scikitml.
    """
    def __init__(self, config_json=None, from_load=False, path=None):
        super().__init__(config_json, from_load, path)
        if from_load == False:
            if config_json is None:
                raise ValueError(f"No config while contstructing model.")
            self.model = None
            self.config = config_json
        else:
            if path is None:
                raise ValueError(f"No config or model paths while contstructing model.")
            model, config = self.load(path)

            self.model = model
            self.config = config

    def fit(self, train):
        self.model.fit(
            X=train.data.loc[:, train.cols_features].values,
            treatment=train.data.loc[:, train.col_treatment].values,
            y=train.data.loc[:, train.col_target].values,
        )

    def predict(self, X: NumpyDataset):           
        scores = X.data.copy(deep=True)
        scores['score'] = self.model.predict(scores.loc[:, X.cols_features])
        return scores[['score', X.col_treatment, X.col_target]]

    def predict_light(self, X: NumpyDataset):
        self.model.predict(X.data.loc[:, X.cols_features])

    def save(self, path_current_setup):
        model_path = os.path.join(path_current_setup, "model.pkl")
        with open(model_path, "wb") as model_file:
            pickle.dump(self.model, model_file)
        config_path = os.path.join(path_current_setup, "config.json")
        with open(config_path, "w") as config_file:
            json.dump(self.config, config_file)

    def load(self, path):
        config_path = path + "/config.json" 
        model_path = path + "/model.pkl"
        if not os.path.exists(config_path):
            raise ValueError(f"No file found at '{config_path}'.")
        if not os.path.exists(model_path):
            raise ValueError(f"No file found at '{model_path}'.")
        
        with open(model_path, 'rb') as f:
            loaded_model = pickle.load(f)
        with open(config_path, 'rb') as f:
            loaded_config = json.load(f)
            
        print(f"Model loaded from {model_path}.")
        print(f"Config loaded from {config_path}.")

        return loaded_model, loaded_config


    def measure_inference_time(self, data, batch_size, max_size=None):

        batches = [
            data[i:i + batch_size]
            for i in range(0, len(data), batch_size)
        ]
    
        if max_size is None:
            max_size = len(data)
    
        inference_times = []
    
        cur_size = 0
        for batch in batches:
            start_time = time.time()
            predictions = self.predict_light(batch)
            end_time = time.time() 
            
            inference_times.append((end_time - start_time) * 1000 / batch_size)
    
            cur_size += batch_size
            if cur_size >= max_size:
                break
    
        mean_inference_time = np.mean(inference_times)
        return mean_inference_time


    @staticmethod
    def generate_config(self, **params):
        pass