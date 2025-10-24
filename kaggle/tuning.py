import os
import tempfile

import ray.cloudpickle as pickle
import torch
import torch.nn as nn
from ray.air import session
from ray.tune import Checkpoint
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np
import xgboost as xgb

from kaggle.models.lightgbm_tree import create_lgbm_model
from kaggle.models.naive_bayes import create_naive_bayes_model
from kaggle.models.neural_network import create_nn_model
from kaggle.models.xgboost_tree import create_xgboost_model


def train_lgbm(config, data):
    X_train, X_val, y_train, y_val = data
    
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    fold_mses = []

    for train_index, val_index in kf.split(X_train, y_train):
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
    
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    fold_mses = []

    for train_index, val_index in kf.split(X_train, y_train):
        X_sub_train, X_sub_val = X_train.iloc[train_index], X_train.iloc[val_index]
        y_sub_train, y_sub_val = y_train.iloc[train_index], y_train.iloc[val_index]

        model = create_xgboost_model(config, X_sub_train, y_sub_train)
        
        dval = xgb.DMatrix(X_sub_val)
        predictions = model.predict(dval)
        mse = mean_squared_error(y_sub_val, predictions)
        fold_mses.append(mse)
    
    avg_mse = np.mean(fold_mses)
    session.report({"mse": avg_mse})

def train_nb(config, data):
    X_train, X_val, y_train, y_val = data
    
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    fold_mses = []

    for train_index, val_index in kf.split(X_train, y_train):
        X_sub_train, X_sub_val = X_train.iloc[train_index], X_train.iloc[val_index]
        y_sub_train, y_sub_val = y_train.iloc[train_index], y_train.iloc[val_index]

        model = create_naive_bayes_model(config, X_sub_train, y_sub_train)
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

    model, optimizer = create_nn_model(
        input_size=input_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        dropout_rate=dropout,
        learning_rate=lr,
        optimizer_name=optimizer_name
    )
    model.to(device)

    train_dataset = TensorDataset(torch.from_numpy(X_train.values).float(), torch.from_numpy(y_train.values).float())
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    val_features = torch.from_numpy(X_val.values).float().to(device)
    val_labels = torch.from_numpy(y_val.values).float().to(device)

    criterion = nn.MSELoss()

    for i in range(50):  # Epoch loop
        model.train()
        total_train_loss = 0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs.squeeze(), labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)

        # Validation at the end of each epoch
        model.eval()
        with torch.no_grad():
            val_outputs = model(val_features)
            val_loss = criterion(val_outputs.squeeze(), val_labels)

        # Collect histograms
        histograms = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                histograms[f"gradients/{name}"] = param.grad.cpu().numpy()
            histograms[f"weights/{name}"] = param.cpu().detach().numpy()

        # Report metrics and checkpoint to Ray Tune
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            with open(os.path.join(checkpoint_dir, "model.pt"), "wb") as f:
                pickle.dump(model.state_dict(), f)
            checkpoint = Checkpoint.from_directory(checkpoint_dir)
            
            session.report(
                {
                    "mse": val_loss.item(),
                    "train_loss": avg_train_loss,
                    "histograms": histograms
                },
                checkpoint=checkpoint
            )
