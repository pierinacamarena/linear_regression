import argparse
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

from batch_gradient_descent import BatchGradientDescent

def standard_scale_data(df: pd.DataFrame) -> pd.DataFrame:
    scaler = StandardScaler()

    numeric_columns = df.select_dtypes(include=[np.number]).columns

    scaled_data = scaler.fit_transform(df[numeric_columns])

    df_scaled = pd.DataFrame(scaled_data, columns=numeric_columns)

    return df_scaled


def read_dataset(file_path: str) -> str:
    """
    Read dataset
    """
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

# def linear_regression(dataset: str) -> any:
#     """Perform linear regression on the dataset""" 
#     theta0 
#     for element in dataset:
    

def main():

    # Parse input arguments 
    parser = argparse.ArgumentParser(description="Receive mileage of a car and dataset path")
    parser.add_argument('mileage', type=int, help="The mileage for a car")
    parser.add_argument('file_path', type=str, help="The filepath for the training dataset")
    
    args = parser.parse_args()

    mileage = args.mileage
    file_path = args.file_path
    
    # Read dataset
    df = read_dataset(file_path)
    
    # Scale dataset
    scaled_data = standard_scale_data(df)
    print(scaled_data)

    independent_var = scaled_data['km']
    target_var = scaled_data['price']
    print(f'independent var is {independent_var}')
    print(f'target var is {target_var}')


    batch_size = scaled_data.shape[0]

    print(batch_size)

    epochs = 10

    for epoch in range(1, epochs + 1):
        print('yo')
        for b in range(0, batch_size, batch_size):
            print(f'b: {b}')
            print(epoch)

    # print(dataset)
    # bgd = BatchGradientDescent(dataset)
    # bgd.linear_regression()

    # print(dataset)
    # print(f'len of dataset is {len(dataset)}')
    # i = 0
    # for element in dataset:
    #     i = i + 1
    # print(f'i is {i}')
if __name__ == "__main__":
    main()