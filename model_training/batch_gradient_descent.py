import numpy as np
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
        return [self.weight * x + self.bias for x in X]


    def predict(self, mileage):
        return self.weight * mileage + self.bias


    def denormalize_coefficients(self, X_scaler: StandardScaler, y_scaler: StandardScaler):

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



    def mean_squared_error(self, y_true: List[float], y_pred: List[float]) -> float:

        if len(y_true) != len(y_pred):
            raise ValueError("The lengths of y_true and y_pred must be equal")
        
        n = len(y_true)
        squared_errors = [(y - y_hat) ** 2 for y, y_hat in zip(y_true, y_pred)]
        mse = sum(squared_errors) / n
        return mse

    def calculate_precission(self) -> None:
        """
        Calculate the precision of the algorithm
        """
        # Predict the prices after training

        y_pred = self.predictArray(self.features)

        # Calculate Mean Squared Error (MSE)
        mse = self.mean_squared_error(self.target, y_pred)
        log.trace(f"""
        Algorithm's precision: [{mse}]
        """)

