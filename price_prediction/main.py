import argparse
import json

def get_coefficients() -> tuple:
    """
    Retrieves the bias (theta0) and weight (theta1) from a JSON file.
    
    Returns:
        tuple: A tuple (theta0, theta1), representing the bias and weight.
        If the file cannot be read or is invalid, returns (0, 0).
    """
    try:
        with open('data/coefficients.json', 'r') as file:
            parsed_data = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError, Exception):
        return (0, 0)

    theta0 = parsed_data.get('bias', 0)
    theta1 = parsed_data.get('weight', 0)

    return (theta0, theta1)


def apply_hypothesis(mileage: int) -> float:

    theta0, theta1 = get_coefficients()

    # Linear equation
    hypothesis = theta0 + (theta1 * mileage)

    return hypothesis


def calculate_price(mileage: int) -> float:

    # Apply the hypothesis to the received mileage
    car_price = apply_hypothesis(mileage)

    return round(car_price, 2)


def main():
    # Parse input arguments 
    parser = argparse.ArgumentParser(description="Receive mileag of a car")
    parser.add_argument('mileage', type=int, help="The mileage for a car")
    args = parser.parse_args()

    mileage = args.mileage
    
    car_price = calculate_price(mileage)
    print(f'Car price for mileage {mileage} is {car_price}')


if __name__ == "__main__":
    main()