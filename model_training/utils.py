import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from batch_gradient_descent import BatchGradientDescent
import matplotlib.pyplot as plt
import logger as logger
log = logger.get_logger()


def standard_scale_data(data: np.array) -> tuple:
    """
    Standardize input data using StandardScaler.

    Args:
        data (np.array): Input data to scale (1D or 2D).

    Returns:
        tuple: (scaled_data, scaler)

    Raises:
        ValueError: If data is empty or non-numeric.
    """
    # Check for empty input
    if data.size == 0:
        raise ValueError("'standard_scale_data' - Input data is empty.")
    
    # Ensure data is numeric
    if not np.issubdtype(data.dtype, np.number):
        raise ValueError("'standard_scale_data' -  Input data contains non-numeric values.")
    
    scaler = StandardScaler()
    
    # Reshape only if data is 1D
    if data.ndim == 1:
        scaled_data = scaler.fit_transform(data.reshape(-1, 1)).flatten()
    else:
        scaled_data = scaler.fit_transform(data)
    
    return scaled_data, scaler


def read_dataset(file_path: str) -> pd.DataFrame:
    """
    Read CSV file, remove NaN rows, and ensure numeric data.

    Args:
        file_path (str): Path to CSV file.

    Returns:
        pd.DataFrame: Cleaned dataset.

    Raises:
        FileNotFoundError: If file doesn't exist.
        ValueError: If data contains non-numeric values.
        Various pandas errors: For parsing issues.
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"'read_dataset' - File not found in: {file_path}")
        df = pd.read_csv(file_path)
        df = df.dropna()

        # Ensure all columns contain numeric data
        if not all(np.issubdtype(df[col].dtype, np.number) for col in df.columns):
            raise ValueError("Input data contains non-numeric values.")
        return df

    except FileNotFoundError as e:
        raise FileNotFoundError(f"'read_dataset' - File not found in: {file_path}") from e
    except pd.errors.EmptyDataError as e:
        raise pd.errors.EmptyDataError(f"'read_dataset' - No data in file in: {file_path}") from e
    except pd.errors.ParserError as e:
        raise pd.errors.ParserError(f"'read_dataset' - Error parsing CSV file in: {file_path}. Error: {str(e)}") from e
    except Exception as e:
        raise Exception(f"'read_dataset' - Unexpected error in for file {file_path}: {str(e)}") from e


def plot_regression_line(X, y, bgd):
    """Plot only the original data points and regression line."""
    # Plot the original data points
    plt.scatter(X, y, color='blue', label='Data Points')

    # Predicted values (regression line)
    y_pred = bgd.predictArray(X)

    # Plot the regression line
    plt.plot(X, y_pred, color='red', label='Regression Line')

    plt.xlabel('Mileage (km)')
    plt.ylabel('Price ($)')
    plt.title('Car Price Prediction')
    plt.legend()
    plt.show()


def plot_original_data(X, y):
    """Plot only the original data points."""
    plt.scatter(X, y, color='blue', label='Data Points')
    plt.xlabel('Mileage (km)')
    plt.ylabel('Price ($)')
    plt.title('Original Car Data (Mileage vs Price)')
    plt.legend()
    plt.show()
