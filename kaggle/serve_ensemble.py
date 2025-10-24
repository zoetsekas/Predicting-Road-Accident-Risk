import os

import joblib
import numpy as np
import torch
from ray import serve

from kaggle.neural_network import NeuralNetwork


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

        # Load Neural Network model
        nn_path = os.path.join(model_dir, "nn_model.pt")
        # Assuming input_size, hidden_size, num_hidden_layers, dropout_rate are known or can be loaded
        # For now, using placeholder values. These should ideally be saved with the model config.
        # In a real scenario, you'd save the NN architecture parameters along with the state_dict.
        input_size = 100 # Placeholder - replace with actual input size from your data
        hidden_size = 64 # Placeholder - replace with actual best config
        num_hidden_layers = 2 # Placeholder - replace with actual best config
        dropout_rate = 0.3 # Placeholder - replace with actual best config

        self.models["nn"] = NeuralNetwork(
            input_size=input_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            dropout_rate=dropout_rate
        )
        self.models["nn"].load_state_dict(torch.load(nn_path))
        self.models["nn"].eval()
        print(f"Loaded Neural Network model from {nn_path}")

    async def __call__(self, request):
        # Assuming the request body contains JSON data with features
        input_data = await request.json()
        features = np.array(input_data["features"])

        # Make predictions with each model
        lgbm_pred = self.models["lgbm"].predict(features)
        xgb_pred = self.models["xgb"].predict(features)

        # Neural Network prediction
        nn_input = torch.tensor(features, dtype=torch.float32)
        with torch.no_grad():
            nn_pred = self.models["nn"](nn_input).numpy()

        # Simple averaging ensemble
        ensemble_pred = (lgbm_pred + xgb_pred + nn_pred.flatten()) / 3

        return {"prediction": ensemble_pred.tolist()}

if __name__ == 'main':
    # To deploy this locally, you would run:
    serve.start()
    EnsembleModel.deploy()
# Then you can query it using:
# curl -X POST -H "Content-Type: application/json" -d '{"features": [[...]]}' http://127.0.0.1:8000/ensemble