import logging
import os
import json

import joblib
import ray
import ray.cloudpickle as pickle
import torch
import numpy as np
import xgboost as xgb
from dotenv import load_dotenv
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch
from sklearn.model_selection import train_test_split

from kaggle.callbacks import ModelLoggerCallback # Import the custom callback
from kaggle.data_preprocessing import (
    load_data,
    clean_data,
    feature_engineer,
    normalize_and_encode_data,
)
from kaggle.tuning import train_lgbm, train_xgb, train_nn, train_nb # Import train_nb
from kaggle.models.linear_regression import create_linear_regression_model
from kaggle.models.neural_network import NeuralNetwork

# Load environment variables from .env file
load_dotenv()

train_data=os.getenv('TRAIN_DATA', '../data/playground-series-s5e10/train.csv')
test_data=os.getenv('TEST_DATA', '../data/playground-series-s5e10/test.csv')

logger = logging.getLogger(__name__)

def processing_pipeline(train_file_path):
    """
    Runs the complete data processing pipeline.

    Args:
        train_file_path (str): The path to the training data CSV file.

    Returns:
        tuple: A tuple containing the processed features (X), target (y), and the scaler object.
    """
    logger.info("Starting data processing pipeline...")
    # 1. Load the data
    df = load_data(train_file_path)

    # 2. Clean the data
    df_cleaned = clean_data(df)

    # 3. Engineer features
    df_featured = feature_engineer(df_cleaned)

    # 4. Normalize and encode the data
    X, y, scaler = normalize_and_encode_data(df_featured)
    logger.info("Data processing pipeline finished.")
    return X, y, scaler

def get_data():
    """Loads and splits the data for training and validation."""
    train_file_path = train_data
    X, y, _ = processing_pipeline(train_file_path)
    return train_test_split(X, y, test_size=0.2, random_state=42)

if __name__ == '__main__':
    # Initialize Ray
    use_cuda = torch.cuda.is_available()
    ray.init(ignore_reinit_error=True, num_cpus=4, num_gpus=1 if use_cuda else 0)

    # Define the path to the training data
    train_file_path = train_data
    
    # Directory to save models
    model_save_dir = "saved_models"
    os.makedirs(model_save_dir, exist_ok=True)

    try:
        # Run the pipeline
        X_train, X_val, y_train, y_val = get_data()

        # Put data in Ray object store to be accessed by all trials
        data_ref = ray.put((X_train, X_val, y_train, y_val))

        # --- Hyperparameter Search Spaces ---
        lgbm_search_space = {
            "objective": "regression_l1",
            "metric": "mse",
            "boosting_type": "gbdt",
            "num_leaves": tune.randint(20, 50),
            "learning_rate": tune.loguniform(1e-4, 1e-1),
            "feature_fraction": tune.uniform(0.7, 1.0),
            "bagging_fraction": tune.uniform(0.7, 1.0),
            "bagging_freq": tune.randint(1, 10),
            "verbose": -1
        }
        if use_cuda:
            lgbm_search_space["device"] = "gpu"

        xgb_search_space = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "eta": tune.loguniform(1e-4, 1e-1),
            "max_depth": tune.randint(3, 10),
            "subsample": tune.uniform(0.5, 1.0),
            "colsample_bytree": tune.uniform(0.5, 1.0),
            "seed": 42
        }
        if use_cuda:
            xgb_search_space["device"] = "cuda"

        nn_search_space = {
            "hidden_size": tune.choice([32, 64, 128]),
            "num_hidden_layers": tune.randint(1, 4),
            "dropout": tune.uniform(0.1, 0.5),
            "lr": tune.loguniform(1e-4, 1e-1),
            "optimizer": tune.choice(["Adam", "SGD"])
        }

        nb_search_space = {
            "var_smoothing": tune.loguniform(1e-9, 1e-2)
        }

        # --- Scheduler ---
        scheduler = ASHAScheduler(metric="mse", mode="min", grace_period=5, reduction_factor=2)

        # --- Run Tuning for Each Model ---
        models_to_tune = {
            "LightGBM": {"trainable": train_lgbm, "space": lgbm_search_space, "filename": "lgbm_model.joblib"},
            "XGBoost": {"trainable": train_xgb, "space": xgb_search_space, "filename": "xgb_model.joblib"},
            "NeuralNetwork": {"trainable": train_nn, "space": nn_search_space, "filename": "nn_model.pt"},
            "NaiveBayes": {"trainable": train_nb, "space": nb_search_space, "filename": "nb_model.joblib"}
        }

        best_models = {}
        for model_name, model_info in models_to_tune.items():
            logger.info(f"--- Tuning {model_name} ---")

            # --- Search Algorithm (re-instantiated for each model) ---
            search_alg = HyperOptSearch(metric="mse", mode="min")

            # --- Custom Logger Callback for each model ---
            callbacks = [ModelLoggerCallback()]

            # The trainable needs to accept the data
            trainable_with_data = tune.with_parameters(model_info["trainable"], data=data_ref)

            analysis = tune.run(
                trainable_with_data,
                name=model_name, # Name the experiment after the model
                resources_per_trial={"cpu": 1, "gpu": 1 if use_cuda else 0},
                config=model_info["space"],
                num_samples=2,  # Number of hyperparameter combinations to try
                search_alg=search_alg,
                scheduler=scheduler,
                stop={"training_iteration": 50},  # Max iterations for each trial
                verbose=1,
                callbacks=callbacks # Pass the custom callback
            )

            best_trial = analysis.get_best_trial(metric="mse", mode="min")
            best_config = best_trial.config
            logger.info(f"Best hyperparameters for {model_name}: {best_config}")

            # Retrain the best model and save it
            logger.info(f"Retraining and saving the best {model_name} model...")
            if model_name == "LightGBM":
                from kaggle.models.lightgbm_tree import create_lgbm_model
                best_model = create_lgbm_model(best_config, X_train, y_train)
                joblib.dump(best_model, os.path.join(model_save_dir, model_info["filename"]))
                best_models[model_name] = best_model
            elif model_name == "XGBoost":
                from kaggle.models.xgboost_tree import create_xgboost_model
                best_model = create_xgboost_model(best_config, X_train, y_train)
                joblib.dump(best_model, os.path.join(model_save_dir, model_info["filename"]))
                best_models[model_name] = best_model
            elif model_name == "NaiveBayes":
                from kaggle.models.naive_bayes import create_naive_bayes_model
                best_model = create_naive_bayes_model(best_config, X_train, y_train)
                joblib.dump(best_model, os.path.join(model_save_dir, model_info["filename"]))
                best_models[model_name] = best_model
            elif model_name == "NeuralNetwork":
                # Add input size to the config
                best_config['input_size'] = X_train.shape[1]
                best_model = NeuralNetwork(
                    input_size=best_config["input_size"],
                    hidden_size=best_config["hidden_size"],
                    num_hidden_layers=best_config["num_hidden_layers"],
                    dropout_rate=best_config["dropout"]
                )
                # Load the best model state from the trial checkpoint
                best_checkpoint = best_trial.checkpoint
                with best_checkpoint.as_directory() as checkpoint_dir:
                    with open(os.path.join(checkpoint_dir, "model.pt"), "rb") as f:
                        best_model.load_state_dict(pickle.load(f))
                torch.save(best_model.state_dict(), os.path.join(model_save_dir, model_info["filename"]))
                best_models[model_name] = best_model
                
                # Save hyperparameters
                hyperparameters_path = os.path.join(model_save_dir, "nn_hyperparameters.json")
                with open(hyperparameters_path, "w") as f:
                    json.dump(best_config, f)
                logger.info(f"Saved Neural Network hyperparameters to {hyperparameters_path}")

            logger.info(f"Best {model_name} model saved to {os.path.join(model_save_dir, model_info['filename'])}")

        # --- Stacking Ensemble ---
        logger.info("--- Creating Stacking Ensemble ---")
        
        # Generate predictions from base models on the validation set
        stacking_features = []
        for model_name, model in best_models.items():
            if model_name == "XGBoost":
                dval = xgb.DMatrix(X_val)
                predictions = model.predict(dval)
            elif model_name == "NeuralNetwork":
                nn_input = torch.tensor(X_val.values, dtype=torch.float32)
                with torch.no_grad():
                    predictions = model(nn_input).numpy().flatten()
            else:
                predictions = model.predict(X_val)
            stacking_features.append(predictions)
        
        stacking_X = np.column_stack(stacking_features)

        # Train the meta-model (Linear Regression)
        meta_model = create_linear_regression_model(stacking_X, y_val)
        
        # Save the meta-model
        meta_model_path = os.path.join(model_save_dir, "meta_model.joblib")
        joblib.dump(meta_model, meta_model_path)
        logger.info(f"Saved meta-model to {meta_model_path}")

    except FileNotFoundError:
        logger.error(f"Error: The file '{train_file_path}' was not found.")
    except Exception as e:
        logger.error(f"An error occurred during the pipeline execution: {e}", exc_info=True)
    finally:
        # Shutdown Ray
        ray.shutdown()
