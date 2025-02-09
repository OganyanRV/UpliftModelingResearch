from src.models.IModelUplift import *

class ICausalML(IModelUplift):
    def __init__(self, config_json=None, from_load=False, path=None):
        super().__init__(config_json, from_load, path)

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

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"Model saved to {path}.")

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