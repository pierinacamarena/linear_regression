import argparse
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from batch_gradient_descent import BatchGradientDescent

def standard_scale_data(data: np.array) -> tuple:
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data.reshape(-1, 1)).flatten()
    return scaled_data, scaler

def read_dataset(file_path: str) -> pd.DataFrame:
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError as e:
        print(f"Error: File not found: {file_path}")
        return None
    except pd.errors.EmptyDataError:
        print(f"Error: No data in file: {file_path}")
        return None
    except pd.errors.ParserError as e:
        print(f"Error: Error parsing CSV file: {file_path}. Error: {str(e)}")
        return None
    except Exception as e:
        print(f"Error: Unexpected error reading file {file_path}: {str(e)}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Train a linear regression model on car data")
    parser.add_argument('file_path', type=str, help="The filepath for the training dataset")
    args = parser.parse_args()

    # Read dataset
    df = read_dataset(args.file_path)
    if df is None:
        return

    # Scale features and target separately
    X = df['km'].values
    y = df['price'].values

    X_scaled, X_scaler = standard_scale_data(X)
    y_scaled, y_scaler = standard_scale_data(y)

    # Train model
    batch_size = len(X_scaled)
    bgd = BatchGradientDescent(X_scaled, y_scaled, batch_size, 5000)

    print(f'Weight before fit: {bgd.weight}')
    print(f'Bias before fit: {bgd.bias}')

    bgd.fit()

    print(f'Weight after fit: {bgd.weight}')
    print(f'Bias after fit: {bgd.bias}')

    # Make predictions
    scaled_predictions = bgd.predict(X_scaled)

    # Denormalize predictions
    denormalized_predictions = y_scaler.inverse_transform(np.array(scaled_predictions).reshape(-1, 1)).flatten()

    print(f'Denormalized predictions: {denormalized_predictions}')
    print(f'Type of denormalized predictions: {type(denormalized_predictions)}')

    # Print some statistics for verification
    print(f'\nOriginal price range: {y.min()} to {y.max()}')
    print(f'Predicted price range: {denormalized_predictions.min()} to {denormalized_predictions.max()}')

if __name__ == "__main__":
    main()