import logging

import pandas as pd
import ray
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

def load_data(file_path):
    """Loads data from a CSV file into a pandas DataFrame."""
    logger.info(f"Loading data from {file_path}")
    df_data = pd.read_csv(file_path)
    return df_data

def clean_data(df):
    """Cleans the data by dropping rows with missing values."""
    logger.info("Cleaning data...")
    return df.dropna()

def feature_engineer(df):
    """
    Engineers new features for the model.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with new features.
    """
    logger.info("Engineering features...")
    # Make a copy to avoid SettingWithCopyWarning
    df = df.copy()

    # Create features from specific categorical columns before they are one-hot encoded
    if 'time_of_day' in df.columns:
        df['is_rush_hour'] = df['time_of_day'].isin(['morning', 'evening']).astype(int)
    if 'weather' in df.columns:
        df['adverse_weather'] = df['weather'].isin(['rainy', 'foggy', 'snowy']).astype(int)
    if 'lighting' in df.columns:
        df['poor_lighting'] = df['lighting'].isin(['night', 'dim']).astype(int)
    if 'adverse_weather' in df.columns and 'poor_lighting' in df.columns:
        df['adverse_conditions'] = ((df['adverse_weather'] == 1) | (df['poor_lighting'] == 1)).astype(int)

    # Interaction features
    if all(c in df.columns for c in ['speed_limit', 'num_lanes', 'curvature']):
        df['speed_per_lane'] = df.apply(lambda row: row['speed_limit'] / row['num_lanes'] if row['num_lanes'] > 0 else 0, axis=1)
        df['curvature_x_lanes'] = df['curvature'] * df['num_lanes']
        df['speed_limit_x_curvature'] = df['speed_limit'] * df['curvature']

    # Polynomial features
    if 'speed_limit' in df.columns:
        df['speed_limit_sq'] = df['speed_limit']**2

    # Binning features
    if 'curvature' in df.columns:
        df['curvature_bin'] = pd.cut(df['curvature'], bins=5, labels=False)
    if 'speed_limit' in df.columns:
        df['speed_limit_bin'] = pd.cut(df['speed_limit'], bins=5, labels=False)

    # Boolean to integer conversion
    for col in ['road_signs_present', 'public_road', 'holiday', 'school_season']:
        if col in df.columns:
            df[col] = df[col].astype(int)

    # One-hot encode all remaining object-type categorical features
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if 'accident_risk' in categorical_cols:
        categorical_cols.remove('accident_risk')
    if categorical_cols:
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)

    return df

def normalize_and_encode_data(df, target_column='accident_risk', scaler=None):
    """
    Normalizes numerical features and separates features from the target.
    Ensures the final feature set is entirely numeric.
    """
    logger.info("Normalizing and encoding data...")
    y = None
    if target_column:
        y = df[target_column]
        X = df.drop(columns=[target_column, 'id'], errors='ignore')
    else:
        X = df.drop(columns=['id'], errors='ignore')

    # At this point, all columns in X should be numeric. Let's verify.
    non_numeric_cols = X.select_dtypes(exclude=['number']).columns
    if not non_numeric_cols.empty:
        raise TypeError(f"Feature engineering left non-numeric columns: {non_numeric_cols.tolist()}")

    # Identify columns for scaling. We exclude binary, one-hot, and binned features.
    cols_to_scale = [
        col for col in X.columns
        if X[col].nunique() > 2 and col not in ['curvature_bin', 'speed_limit_bin']
    ]

    if scaler is None:
        scaler = StandardScaler()
        if cols_to_scale:
            X[cols_to_scale] = scaler.fit_transform(X[cols_to_scale])
    else:
        if cols_to_scale:
            X[cols_to_scale] = scaler.transform(X[cols_to_scale])

    return X, y, scaler

if __name__ == '__main__':
    ray.init(ignore_reinit_error=True)
    file_path = 'data/playground-series-s5e10/train.csv'
    try:
        df = load_data(file_path)
        df_cleaned = clean_data(df)
        df_featured = feature_engineer(df_cleaned)
        
        X, y, scaler = normalize_and_encode_data(df_featured)

        logger.info("Data loaded, cleaned, features engineered, and normalized successfully.")
        logger.info(f"Original shape: {df.shape}")
        logger.info(f"Cleaned shape: {df_cleaned.shape}")
        logger.info(f"Featured shape: {df_featured.shape}")
        logger.info(f"Features (X) shape: {X.shape}")
        logger.info(f"Target (y) shape: {y.shape}")
        logger.info("\nFirst 5 rows of the processed Features (X):\n%s", X.head())
        logger.info("\nFirst 5 rows of the Target (y):\n%s", y.head())
        logger.info("\nColumns of the processed Features (X):\n%s", X.columns.tolist())

    except FileNotFoundError:
        logger.error(f"Error: The file '{file_path}' was not found.")
        logger.info("Creating a dummy dataframe for demonstration.")
    finally:
        ray.shutdown()
