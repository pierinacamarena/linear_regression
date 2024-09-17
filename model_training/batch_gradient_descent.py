from typing import Optional
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
