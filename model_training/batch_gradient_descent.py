import numpy as np
import json
from typing import Optional, List
from sklearn.preprocessing import StandardScaler

import logger as logger
log = logger.get_logger()


class BatchGradientDescent:

    def __init__(self, features , target, batch_size, epochs = 100):
        self.learning_rate = 0.01
        self.bias = 0
        self.weight = 0
        self.features = features
        self.target = target
        self.batch_size = batch_size
        self.epochs = epochs
        log.trace(f"""
        BEFORE TRAINING:
            weight: [{self.weight}]
            bias: [{self.bias}]
        """)


    def fit(self) -> None :
        """
        Train the linear regression model using gradient descent.
        """
        for _ in range(self.epochs):
            y_predicted = [self.weight * x + self.bias for x in self.features]

            # Compute gradients
            dw = sum((y_predicted[i] - self.target[i]) * self.features[i] for i in range(self.batch_size)) / self.batch_size
            db = sum(y_predicted[i] - self.target[i] for i in range(self.batch_size)) / self.batch_size

            tempWeight = self.learning_rate * dw
            tempBias =  self.learning_rate * db
            
            # Update coefficients
            self.weight -= tempWeight
            self.bias -= tempBias

        log.trace(f"""
        AFTER TRAINING:
            weight: [{self.weight}]
            bias: [{self.bias}]
        """)


    def predictArray(self, X):
        """
        Predict outputs for an array of input features.

        Args:
            X: Array-like of input features.

        Returns:
            List of predicted values.
        """
        return [self.weight * x + self.bias for x in X]


    def predict(self, mileage):
        """
        Predict output for a single input value.

        Args:
            mileage: Single input value.

        Returns:
            Predicted output value.
        """
        prediction = self.weight * mileage + self.bias
        if prediction < 0:
            return 0
        return prediction

    def denormalize_coefficients(self, X_scaler: StandardScaler, y_scaler: StandardScaler):
        """
        Denormalize model coefficients using provided scalers.

        Args:
            X_scaler: StandardScaler for features.
            y_scaler: StandardScaler for target.
        """
        # Denormalize the weight
        denormalized_weight = (self.weight * y_scaler.scale_[0]) / X_scaler.scale_[0]

        # Denormalize the bias
        denormalized_bias = y_scaler.mean_[0] + (self.bias * y_scaler.scale_[0]) - \
                            (X_scaler.mean_[0] * denormalized_weight)

        self.weight = denormalized_weight
        self.bias = denormalized_bias

        log.trace(f"""
        AFTER DENORMALIZATION:
            weight: [{self.weight}]
            bias: [{self.bias}]
        """)

    def save_coefficients(self):
        data = {
            'weight': self.weight,
            'bias': self.bias
        }
        with open('data/coefficients.json', 'w') as file:
            json.dump(data, file)


    def mean_squared_error(self, y_true: List[float], y_pred: List[float]) -> float:
        """
        Calculate mean squared error between true and predicted values.

        Args:
            y_true: List of true values.
            y_pred: List of predicted values.

        Returns:
            Mean squared error.

        Raises:
            ValueError: If input lists have invalid lengths.
        """

        if len(y_true) != len(y_pred):
            raise ValueError("mean_squared_error - The lengths of y_true and y_pred must be equal")
        n = len(y_true)
        if n < 1:
            raise ValueError("mean_squared_error - The lengths of y_true and y_pred must be greater than 0")

        squared_errors = [(y - y_hat) ** 2 for y, y_hat in zip(y_true, y_pred)]
        mse = sum(squared_errors) / n
        return mse


    def calculate_precission(self) -> None:
        """
        Calculate and log the precision (MSE) of the model.
        """
        # Predict the prices after training

        y_pred = self.predictArray(self.features)

        # Calculate Mean Squared Error (MSE)
        mse = self.mean_squared_error(self.target, y_pred)
        log.trace(f"""
        Algorithm's precision: [{round(mse, 2)}]
        """)

