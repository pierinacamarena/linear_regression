import argparse
import pandas as pd
import numpy as np
from batch_gradient_descent import BatchGradientDescent
from utils import standard_scale_data, read_dataset, plot_regression_line, plot_original_data
import logger as logger
log = logger.get_logger()

def main():
    parser = argparse.ArgumentParser(description="Train a linear regression model on car data")
    parser.add_argument('file_path', type=str, help="The filepath for the training dataset")

    # Optional argument for mileage with default value of None
    parser.add_argument('--mileage', type=int, default=None, help="A mileage to predict a car price (default: 0)")
    
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
    log.info('Instanting the Batch Gradient Descent Class')
    bgd = BatchGradientDescent(X_scaled, y_scaled, len(X_scaled), 5000)

    if train_model:
        log.info('Training the model')
        bgd.fit()

        log.info("Calculating model's precision")
        bgd.calculate_precission()

        log.info("Denormalizing coefficients")
        bgd.denormalize_coefficients(X_scaler, y_scaler)

        log.info("Plotting the regression line")
        plot_regression_line(X, y, bgd)


    if mileage or mileage == 0:
        log.info(f"Calculating the price for mileage [{mileage}]")
        predicted_price = round(bgd.predict(mileage), 2)
        log.trace(f'''
        Predicted Price: [{predicted_price}]
        ''')


if __name__ == "__main__":
    main()