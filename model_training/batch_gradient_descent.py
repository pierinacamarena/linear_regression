from typing import Optional, List
import numpy as np

class BatchGradientDescent:

    def __init__(self, features , target, batch_size, epochs = 100):
        self.learning_rate = 0.01
        self.bias = 0
        self.weight = 0
        self.features = features
        self.target = target
        self.batch_size = batch_size
        self.epochs = epochs


    def fit(self) -> None :

        for _ in range(self.epochs):
            y_predicted = [self.weight * x + self.bias for x in self.features]

            # Compute gradients
            dw = sum((y_predicted[i] - self.target[i]) * self.features[i] for i in range(self.batch_size)) / self.batch_size
            db = sum(y_predicted[i] - self.target[i] for i in range(self.batch_size)) / self.batch_size

            # Update coefficients
            self.weight -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predictArray(self, X):
        return [self.weight * x + self.bias for x in X]

    def predict(self, mileage):
        return self.weight * mileage + self.bias
    
    def mean_squared_error(self, y_true: List[float], y_pred: List[float]) -> float:

        if len(y_true) != len(y_pred):
            raise ValueError("The lengths of y_true and y_pred must be equal")
        
        n = len(y_true)
        squared_errors = [(y - y_hat) ** 2 for y, y_hat in zip(y_true, y_pred)]
        mse = sum(squared_errors) / n
        return mse