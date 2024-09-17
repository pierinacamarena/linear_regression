import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from batch_gradient_descent import BatchGradientDescent
import matplotlib.pyplot as plt



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

def denormalize_coefficients(bgd: BatchGradientDescent, X_scaler: StandardScaler,  y_scaler: StandardScaler) -> (float, float):
    
        # Denormalize the weight
        denormalized_weight = (bgd.weight * y_scaler.scale_[0]) / X_scaler.scale_[0]

        # Denormalize the bias
        denormalized_bias = y_scaler.mean_[0] + (bgd.bias * y_scaler.scale_[0]) - \
                            (X_scaler.mean_[0] * denormalized_weight)

        return (denormalized_weight, denormalized_bias)

def plot_regression_line(X, y, bgd):
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
