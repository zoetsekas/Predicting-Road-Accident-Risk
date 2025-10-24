import os
import json

import joblib
import numpy as np
import torch
import xgboost as xgb # Import xgboost
from ray import serve

from kaggle.models.neural_network import NeuralNetwork


@serve.deployment(num_replicas=1, route_prefix="/ensemble")
class EnsembleModel:
    def __init__(self):
        self.models = {}
        model_dir = "saved_models"

        # Load LightGBM model
        lgbm_path = os.path.join(model_dir, "lgbm_model.joblib")
        self.models["lgbm"] = joblib.load(lgbm_path)
        print(f"Loaded LightGBM model from {lgbm_path}")

        # Load XGBoost model
        xgb_path = os.path.join(model_dir, "xgb_model.joblib")
        self.models["xgb"] = joblib.load(xgb_path)
        print(f"Loaded XGBoost model from {xgb_path}")

        # Load Naive Bayes model
        nb_path = os.path.join(model_dir, "nb_model.joblib")
        self.models["nb"] = joblib.load(nb_path)
        print(f"Loaded Naive Bayes model from {nb_path}")

        # Load Neural Network model
        nn_path = os.path.join(model_dir, "nn_model.pt")
        hyperparameters_path = os.path.join(model_dir, "nn_hyperparameters.json")

        with open(hyperparameters_path, "r") as f:
            nn_hyperparameters = json.load(f)
        print(f"Loaded Neural Network hyperparameters from {hyperparameters_path}")

        self.models["nn"] = NeuralNetwork(
            input_size=nn_hyperparameters["input_size"],
            hidden_size=nn_hyperparameters["hidden_size"],
            num_hidden_layers=nn_hyperparameters["num_hidden_layers"],
            dropout_rate=nn_hyperparameters["dropout"]
        )
        self.models["nn"].load_state_dict(torch.load(nn_path))
        self.models["nn"].eval()
        print(f"Loaded Neural Network model from {nn_path}")

        # Load the meta-model
        meta_model_path = os.path.join(model_dir, "meta_model.joblib")
        self.meta_model = joblib.load(meta_model_path)
        print(f"Loaded meta-model from {meta_model_path}")

    async def __call__(self, request):
        # Assuming the request body contains JSON data with features
        input_data = await request.json()
        features = np.array(input_data["features"])

        # Make predictions with each base model
        lgbm_pred = self.models["lgbm"].predict(features)
        
        # Convert features to DMatrix for XGBoost
        dfeatures = xgb.DMatrix(features)
        xgb_pred = self.models["xgb"].predict(dfeatures)
        
        nb_pred = self.models["nb"].predict(features)

        # Neural Network prediction
        nn_input = torch.tensor(features, dtype=torch.float32)
        with torch.no_grad():
            nn_pred = self.models["nn"](nn_input).numpy().flatten()

        # Create the input for the meta-model
        stacking_X = np.column_stack((lgbm_pred, xgb_pred, nn_pred, nb_pred))

        # Get the final prediction from the meta-model
        final_prediction = self.meta_model.predict(stacking_X)

        return {"prediction": final_prediction.tolist()}

if __name__ == 'main':
    # To deploy this locally, you would run:
    serve.start()
    EnsembleModel.deploy()
    # Then you can query it using:
    # curl -X POST -H "Content-Type: application/json" -d '{"features": [[...]]}' http://127.0.0.1:8000/ensemble
