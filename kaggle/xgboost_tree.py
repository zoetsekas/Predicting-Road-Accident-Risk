import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold


def create_xgboost_model(params, X_train, y_train):
    """
    Creates and trains an XGBoost model with the given parameters.

    Args:
        params (dict): A dictionary of XGBoost parameters.
        X_train (pd.DataFrame or np.array): Training data features.
        y_train (pd.Series or np.array): Training data labels.

    Returns:
        xgb.Booster: The trained XGBoost model.
    """
    dtrain = xgb.DMatrix(X_train, label=y_train)
    model = xgb.train(params, dtrain)
    return model

if __name__ == '__main__':
    # This is an example of how to use the create_xgboost_model function with StratifiedKFold.
    # You would replace this with your actual data loading and preprocessing.

    # Create dummy data for demonstration
    X = pd.DataFrame(np.random.rand(100, 5), columns=[f'feature_{i}' for i in range(5)])
    y = pd.Series(np.random.randint(0, 2, 100))

    # Define hyperparameters for the XGBoost model
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'eta': 0.1,
        'max_depth': 3,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'seed': 42
    }

    # Initialize StratifiedKFold
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    models = []
    for fold, (train_index, val_index) in enumerate(skf.split(X, y)):
        print(f"--- Fold {fold+1}/{n_splits} ---")
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        # Create and train the model
        model = create_xgboost_model(params, X_train, y_train)
        models.append(model)

        print(f"Model for fold {fold+1} created successfully.")

    print(f"\n{n_splits}-fold cross-validation finished. {len(models)} models created.")

    # You can then use these models for prediction on a test set, for example by averaging their predictions.
