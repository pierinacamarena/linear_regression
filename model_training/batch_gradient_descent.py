from typing import Optional
class BatchGradientDescent:

    def __init__(self, dataset : str, ):
        self.learning_rate = 0.01
        self.theta0 = 0
        self.theta1 = 0
        self.dataset = dataset[1:]


    def linear_regression(self) -> None :
        m = len(self.dataset)
        # 4th entry
        data = self.dataset[3]
        price, mileage = self.get_price_and_mileage(data)
        print(f'price is: {price}')
        print(f'mileage is: {mileage}')
        # for data in self.dataset:
        #     price, mileage = self.get_price_and_mileage(data)
        #     print(f'data is {data}')
        #     print(f'price is: {price}')
        #     print(f'mileage is: {mileage}')
        #     estimated_price = self.apply_hypothesis(mileage)

    def apply_hypothesis(self, mileage: int) -> float:
    
        # Linear equation
        hypothesis = self.theta0 + (self.theta1 * mileage)

        return hypothesis
    
    # def calculate_value_theta(self, price: int, mileage: Optional[int] = None) -> float:

    def calculate_sum(self, entry: int) -> any:
        sum = 0
        for i in range(entry):
            sum += 1

    # def calculate_value_theta0(self, price: int, estimated_price: float, m: int) -> float:
    #     self.theta0 = self.learning_rate * 1/m 


    def get_price_and_mileage(self, dataset_line: str) -> (int, int):
        result = tuple(map(int, dataset_line.strip().split(',')))
        return result

