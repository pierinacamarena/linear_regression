import argparse
import pandas as pd
import numpy as np
from batch_gradient_descent import BatchGradientDescent
from utils import standard_scale_data, read_dataset, denormalize_coefficients, plot_regression_line, plot_original_data
import logger as logger
log = logger.get_logger()

def main():
    parser = argparse.ArgumentParser(description="Train a linear regression model on car data")
    parser.add_argument('file_path', type=str, help="The filepath for the training dataset")

    # Optional argument for mileage with default value of 0
    parser.add_argument('--mileage', type=int, default=0, help="A mileage to predict a car price (default: 0)")
    
    # Optional argument for train with default value of False
    parser.add_argument('--train', type=bool, default=False, help="Enter True if you want to train the model (default: False)")
    
    args = parser.parse_args()
    
    train_model = args.train
    mileage = args.mileage

    # Read dataset
    df = read_dataset(args.file_path)
    if df is None:
        log.error(f'Unable to retrieve the df')
        return

    # Scale features and target separately
    X = df['km'].values
    y = df['price'].values

    # Plot the original data before any training
    plot_original_data(X, y)

    X_scaled, X_scaler = standard_scale_data(X)
    y_scaled, y_scaler = standard_scale_data(y)

    # Train model
    batch_size = len(X_scaled)
    bgd = BatchGradientDescent(X_scaled, y_scaled, batch_size, 5000)
    log.trace('Instantiated the Batch Gradient Descent')

    log.info(f'Weight before fit: {bgd.weight}')
    log.info(f'Bias before fit: {bgd.bias}')

    if train_model:
        log.info('Training the model')

        bgd.fit()

        log.trace(f'Weight after fit: {bgd.weight}')
        log.trace(f'Bias after fit: {bgd.bias}')

        # Predict the prices after training

        y_pred = bgd.predictArray(X_scaled)

        # Calculate Mean Squared Error (MSE)
        mse = bgd.mean_squared_error(y_scaled, y_pred)
        log.trace(f'Mean Squared Error: {mse}')

        bgd.weight, bgd.bias = denormalize_coefficients(bgd, X_scaler, y_scaler)

        log.info(f'Denormalized Weight: {bgd.weight}')
        log.info(f'Denormalized Bias: {bgd.bias}')

        # Plot the regression line after training
        plot_regression_line(X, y, bgd)


    if mileage:
        predicted_price = round(bgd.predict(mileage), 2)
        log.trace(f'Predicted Price for mileage {mileage}: {predicted_price}')


if __name__ == "__main__":
    main()