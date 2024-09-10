import argparse

def read_dataset() -> str:
    """
    Read dataset
    """
    try:
        # Using with automatically closes the file after the block is executed, even if an error occurs.
        with open("data/data.csv", "r") as dataset:
            dataset = dataset.read()
            print(f'dataset is {dataset}')
            return dataset
    except Exception as e:
        print(f'Error: Opening file data.csv - {e}')
        return ''


def main():

    # Parse input arguments 
    parser = argparse.ArgumentParser(description="Receive mileag of a car")
    parser.add_argument('mileage', type=int, help="The mileage for a car")
    args = parser.parse_args()

    mileage = args.mileage
    print(mileage)
    read_dataset()

if __name__ == "__main__":
    main()