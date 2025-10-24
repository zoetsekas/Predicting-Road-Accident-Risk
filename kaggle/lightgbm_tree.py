import logging

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)

def create_lgbm_model(params, X_train, y_train):
    """
    Creates and trains a LightGBM model with the given parameters.

    Args:
        params (dict): A dictionary of LightGBM parameters.
        X_train: Training data features.
        y_train: Training data labels.

    Returns:
        lgb.Booster: The trained LightGBM model.
    """
    logger.info("Creating LightGBM model...")
    lgb_train = lgb.Dataset(X_train, y_train)
    model = lgb.train(params, lgb_train)
    logger.info("LightGBM model created.")
    return model

if __name__ == '__main__':
    # This is an example of how to use the create_lgbm_model function with StratifiedKFold.
    # You would replace this with your actual data loading and preprocessing.

    # Create dummy data for demonstration
    X = pd.DataFrame(np.random.rand(100, 5), columns=[f'feature_{i}' for i in range(5)])
    y = pd.Series(np.random.randint(0, 2, 100))

    # Define hyperparameters for the LightGBM model
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1
    }

    # Initialize StratifiedKFold
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    models = []
    for fold, (train_index, val_index) in enumerate(skf.split(X, y)):
        logger.info(f"--- Fold {fold+1}/{n_splits} ---")
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        # Create and train the model
        model = create_lgbm_model(params, X_train, y_train)
        models.append(model)

        logger.info(f"Model for fold {fold+1} created successfully.")

    logger.info(f"\n{n_splits}-fold cross-validation finished. {len(models)} models created.")

    # You can then use these models for prediction on a test set, for example by averaging their predictions.
