import os
import tempfile

import ray.cloudpickle as pickle
import torch
import torch.nn as nn
from ray.air import session
from ray.tune import Checkpoint
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold # Changed to StratifiedKFold
from sklearn.metrics import mean_squared_error
import numpy as np

from kaggle.lightgbm_tree import create_lgbm_model
from kaggle.neural_network import create_model as create_nn_model
from kaggle.xgboost_tree import create_xgboost_model


def train_lgbm(config, data):
    X_train, X_val, y_train, y_val = data
    
    # Using StratifiedKFold as requested. Note: This assumes y_train is categorical or binned.
    # If y_train is continuous, KFold should be used instead, or y_train needs to be binned.
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42) 
    fold_mses = []

    # Ensure y_train is suitable for StratifiedKFold (e.g., integer labels)
    # If y_train is continuous, this will raise an error.
    # A common workaround for continuous y is to bin it for stratification,
    # or simply use KFold.
    
    for train_index, val_index in skf.split(X_train, y_train): # StratifiedKFold needs y for splitting
        X_sub_train, X_sub_val = X_train.iloc[train_index], X_train.iloc[val_index]
        y_sub_train, y_sub_val = y_train.iloc[train_index], y_train.iloc[val_index]

        model = create_lgbm_model(config, X_sub_train, y_sub_train)
        predictions = model.predict(X_sub_val)
        mse = mean_squared_error(y_sub_val, predictions)
        fold_mses.append(mse)
    
    avg_mse = np.mean(fold_mses)
    session.report({"mse": avg_mse})

def train_xgb(config, data):
    X_train, X_val, y_train, y_val = data
    
    # Using StratifiedKFold as requested. Note: This assumes y_train is categorical or binned.
    # If y_train is continuous, KFold should be used instead, or y_train needs to be binned.
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42) 
    fold_mses = []

    for train_index, val_index in skf.split(X_train, y_train): # StratifiedKFold needs y for splitting
        X_sub_train, X_sub_val = X_train.iloc[train_index], X_train.iloc[val_index]
        y_sub_train, y_sub_val = y_train.iloc[train_index], y_train.iloc[val_index]

        model = create_xgboost_model(config, X_sub_train, y_sub_train)
        predictions = model.predict(X_sub_val)
        mse = mean_squared_error(y_sub_val, predictions)
        fold_mses.append(mse)
    
    avg_mse = np.mean(fold_mses)
    session.report({"mse": avg_mse})

def train_nn(config, data):
    X_train, X_val, y_train, y_val = data
    
    input_size = X_train.shape[1]
    hidden_size = config["hidden_size"]
    num_hidden_layers = config["num_hidden_layers"]
    dropout = config["dropout"]
    lr = config["lr"]
    optimizer_name = config["optimizer"]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Using StratifiedKFold as requested. Note: This assumes y_train is categorical or binned.
    # If y_train is continuous, KFold should be used instead, or y_train needs to be binned.
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42) 
    fold_mses = []

    for fold_idx, (train_index, val_index) in enumerate(skf.split(X_train, y_train)): # StratifiedKFold needs y for splitting
        X_sub_train, X_sub_val = X_train.iloc[train_index], X_train.iloc[val_index]
        y_sub_train, y_sub_val = y_train.iloc[train_index], y_train.iloc[val_index]

        model, optimizer = create_nn_model(
            input_size=input_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            dropout_rate=dropout,
            learning_rate=lr,
            optimizer_name=optimizer_name
        )
        model.to(device)

        train_dataset = TensorDataset(torch.from_numpy(X_sub_train.values).float(), torch.from_numpy(y_sub_train.values).float())
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        criterion = nn.MSELoss()

        for i in range(50):  # Number of epochs
            for batch_idx, (features, labels) in enumerate(train_loader):
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs.squeeze(), labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
        # Validation loss for the current fold
        val_features = torch.from_numpy(X_sub_val.values).float().to(device)
        val_labels = torch.from_numpy(y_sub_val.values).float().to(device)
        val_outputs = model(val_features)
        val_loss = criterion(val_outputs.squeeze(), val_labels)
        fold_mses.append(val_loss.item())
        
    avg_mse = np.mean(fold_mses)
    
    # Save checkpoint and report metrics (using the last trained model for checkpoint, but average MSE for reporting)
    with tempfile.TemporaryDirectory() as checkpoint_dir:
        with open(os.path.join(checkpoint_dir, "model.pt"), "wb") as f:
            pickle.dump(model.state_dict(), f)
        checkpoint = Checkpoint.from_directory(checkpoint_dir)
        session.report({"mse": avg_mse}, checkpoint=checkpoint)
