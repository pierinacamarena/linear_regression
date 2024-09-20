import argparse
import pandas as pd
import numpy as np
from batch_gradient_descent import BatchGradientDescent
from utils import standard_scale_data, read_dataset, plot_regression_line, plot_original_data
import logger as logger
log = logger.get_logger()


def parse_arguments() -> argparse.Namespace :
    """
    Parse command line arguments for linear regression model.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Train a linear regression model on car data")
    parser.add_argument('file_path', type=str, help="The filepath for the training dataset")

    # Optional argument for mileage with default value of None
    parser.add_argument('--mileage', type=int, default=None, help="A mileage to predict a car price (default: 0)")
    
    # Optional argument for train with default value of False
    parser.add_argument('--train', type=bool, default=False, help="Enter True if you want to train the model (default: False)")
    
    # Optional argument for train with default value of False
    parser.add_argument('--bonus', type=bool, default=False, help="Enter True if you want to show the bonus (default: False)")

    args = parser.parse_args()

    return args


def get_dataset(file_path: str) -> (np.array, np.array):
    """
    Load and prepare dataset from CSV file.

    Args:
        file_path (str): Path to CSV file.

    Returns:
        tuple: (X, y) arrays for features and target.
    """

    # Read dataset
    df = read_dataset(file_path)

    # Scale features and target separately
    X = df['km'].values
    y = df['price'].values

    return (X, y)


def main():

    try:
        # Parse arguments
        args = parse_arguments()
        
        train_model = args.train
        mileage = args.mileage
        file_path = args.file_path
        bonus = args.bonus

        X,y = get_dataset(file_path)

        if bonus:
            # Plot the original data before any training
            plot_original_data(X, y)

        # Standardize the data
        X_scaled, X_scaler = standard_scale_data(X)
        y_scaled, y_scaler = standard_scale_data(y)

        # Instantiate Batch Gradient Descent Class
        log.info('Instanting the Batch Gradient Descent Class')
        bgd = BatchGradientDescent(X_scaled, y_scaled, len(X_scaled), 5000)

    except Exception as e:
        log.error(f'Error: {e}')
        return

    if train_model:
        try: 
            log.info('Training the model')
            bgd.fit()

            if bonus:
                log.info("Calculating model's precision")
                bgd.calculate_precission()

            log.info("Denormalizing coefficients")
            bgd.denormalize_coefficients(X_scaler, y_scaler)

            if bonus: 
                # Plotting the regression line
                plot_regression_line(X, y, bgd)
        except Exception as e:
            log.error(f'Error: {e}')
            return

    if mileage or mileage == 0:
        try: 
            log.info(f"Calculating the price for mileage [{mileage}]")
            predicted_price = round(bgd.predict(mileage), 2)
            log.trace(f'''
            Predicted Price: [{predicted_price}]
            ''')
        except Exception as e:
            log.error(f'Error: {e}')
            return


if __name__ == "__main__":
    main()