import argparse


def apply_hypothesis(mileage: int) -> float:

    # Set theta0 and theta1 to 0
    theta0 = 0
    theta1 = 0

    # Linear equation
    hypothesis = theta0 + (theta1 * mileage)

    return hypothesis

def calculate_price(mileage: int) -> float:
    car_price = apply_hypothesis(mileage)

    return car_price

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