import pandas as pd
import requests

from kaggle.data_preprocessing import (
    load_data,
    clean_data,
    feature_engineer,
    normalize_and_encode_data,
)
from kaggle.pipeline import processing_pipeline


def process_test_data(test_file_path, scaler):
    """
    Processes the test data using the same pipeline as the training data.
    """
    df_test = load_data(test_file_path)
    test_ids = df_test['id']
    df_cleaned = clean_data(df_test)
    df_featured = feature_engineer(df_cleaned)
    X_test, _, _ = normalize_and_encode_data(df_featured, target_column=None, scaler=scaler)
    return X_test, test_ids

if __name__ == '__main__':
    # 1. Get the scaler from the training pipeline
    train_file_path = '../data/playground-series-s5e10/train.csv'
    _, _, scaler = processing_pipeline(train_file_path)

    # 2. Process test data
    test_file_path = '../data/playground-series-s5e10/test.csv'
    X_test, test_ids = process_test_data(test_file_path, scaler)

    # 3. Make predictions
    predictions = []
    for _, row in X_test.iterrows():
        payload = {"features": [row.tolist()]}
        try:
            response = requests.post("http://127.0.0.1:8000/ensemble", json=payload)
            response.raise_for_status()  # Raise an exception for bad status codes
            predictions.append(response.json()["prediction"][0])
        except requests.exceptions.RequestException as e:
            print(f"Error making prediction for row: {e}")
            # Handle error, e.g., by appending a default value or breaking the loop
            break

    # 4. Create submission file
    if len(predictions) == len(test_ids):
        submission_df = pd.DataFrame({'id': test_ids, 'accident_risk': predictions})
        submission_df.to_csv('submission.csv', index=False)
        print("Submission file created successfully: submission.csv")
    else:
        print("Could not create submission file due to prediction errors.")
