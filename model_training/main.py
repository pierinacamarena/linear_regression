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

    # Optional argument for mileage with default value of 0
    parser.add_argument('--mileage', type=int, default=0, help="A mileage to predict a car price (default: 0)")
    
    # Optional argument for train with default value of False
    parser.add_argument('--train_model', type=bool, default=False, help="Enter True if you want to train the model (default: False)")
    
    args = parser.parse_args()
    
    train_model = args.train_model
    mileage = args.mileage

    # Example of how you might use the arguments
    print(f"File Path: {args.file_path}")
    print(f"Mileage: {args.mileage}")
    print(f"Train: {args.train_model}")


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

    if train_model:

        bgd.fit()

        print(f'Weight after fit: {bgd.weight}')
        print(f'Bias after fit: {bgd.bias}')

        # Denormalize the weight
        bgd.weight = (bgd.weight * y_scaler.scale_[0]) / X_scaler.scale_[0]

        # Denormalize the bias
        bgd.bias = y_scaler.mean_[0] + (bgd.bias * y_scaler.scale_[0]) - \
                            (X_scaler.mean_[0] * bgd.weight)

        print(f'Denormalized Weight: {bgd.weight}')
        print(f'Denormalized Bias: {bgd.bias}')

    # Now use the denormalized coefficients for predictions
    if mileage:
        # predicted_price = denormalized_weight * mileage + denormalized_bias
        predicted_price = bgd.predict(mileage)
        print(f'Predicted Price for mileage {mileage}: {predicted_price}')


if __name__ == "__main__":
    main()