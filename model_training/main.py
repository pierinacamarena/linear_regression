import argparse
import pandas as pd
import numpy as np
from batch_gradient_descent import BatchGradientDescent
from utils import standard_scale_data, read_dataset, denormalize_coefficients


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

        bgd.weight, bgd.bias = denormalize_coefficients(bgd, X_scaler, y_scaler)

        print(f'Denormalized Weight: {bgd.weight}')
        print(f'Denormalized Bias: {bgd.bias}')

    if mileage:
        predicted_price = round(bgd.predict(mileage), 2)
        print(f'Predicted Price for mileage {mileage}: {predicted_price}')


if __name__ == "__main__":
    main()