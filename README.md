# Linear Regression Model for Car Price Prediction

This program implements a linear regression model to predict car prices based on mileage using batch gradient descent, for the AI specialization of Ecole42

## Features

- Train a linear regression model on car data
- Predict car prices for given mileages
- Visualize original data and regression line (bonus feature)
- Calculate model precision (bonus feature)

## Prerequisites

- Python 3.10
- Required libraries:
  - colorlog==6.8.2
  - matplotlib==3.8.2
  - numpy==2.1.1
  - pandas==2.2.2
  - scikit_learn==1.3.2

## Installation

### Using Conda (Recommended)

1. Clone this repository:
git clone [repository-url]
Copy
2. Navigate to the project directory:
cd linear-regression
Copy
3. Create a new Conda environment:
conda create -n linear-regression python=3.10
Copy
4. Activate the environment:
conda activate linear-regression
Copy
5. Install the required packages:
conda install -c conda-forge colorlog=6.8.2 matplotlib=3.8.2 numpy=2.1.1 pandas=2.2.2 scikit-learn=1.3.2

### Using pip

If you prefer not to use Conda, you can install the dependencies using pip:

1. Follow steps 1-2 from the Conda instructions above.

2. Install the required packages:
pip install -r requirements.txt

## Usage

Run the program using the following command:

- For price prediction: python -m price_prediction.main [file_path]
- For model training: python -m model_training.main --options [file_path]

### Arguments:

- `file_path`: Path to the CSV file containing the dataset (required)

### Options:

- `--mileage`: Mileage to predict a car price (default: None)
- `--train`: Set to True to train the model (default: False)
- `--bonus`: Set to True to show bonus visualizations (default: False)

### Examples:

1. Predict price for a specific mileage:
- python -m price_prection.main 8900

2. Train the model
- python model_training/main.py --train True data/car_data.csv

3. Train the model and get the prediction for a specific mileage, without bonus :
- python model_training/main.py --mileage 8900 --train True  data/car_data.csv

4. Train the model and get the prediction for a specific mileage, wiht bonus visualizations:
- python model_training/main.py --mileage 8900 --train True --bonus True data/car_data.csv


## File Structure

- `main.py`: Main script to run the program
- `batch_gradient_descent.py`: Contains the BatchGradientDescent class
- `utils.py`: Utility functions for data processing and visualization
- `logger.py`: Custom logging configuration


### Documentation

- https://www.youtube.com/watch?v=4b4MUYve_U8
- https://www.geeksforgeeks.org/what-is-standardscaler/
- https://stackoverflow.com/questions/40758562/can-anyone-explain-me-standardscaler
- https://www.digitalocean.com/community/tutorials/standardscaler-function-in-python
