import argparse

def read_dataset() -> any:



def main():
    # Parse input arguments 
    parser = argparse.ArgumentParser(description="Receive mileag of a car")
    parser.add_argument('mileage', type=int, help="The mileage for a car")
    args = parser.parse_args()

    mileage = args.mileage

if __name__ == "__main__":
    main()