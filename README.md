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
Copy
### Using pip

If you prefer not to use Conda, you can install the dependencies using pip:

1. Follow steps 1-2 from the Conda instructions above.

2. Install the required packages:
pip install -r requirements.txt
Copy
## Usage

Run the program using the following command:
python main.py [file_path] [options]
Copy
### Arguments:

- `file_path`: Path to the CSV file containing the dataset (required)

### Options:

- `--mileage`: Mileage to predict a car price (default: None)
- `--train`: Set to True to train the model (default: False)
- `--bonus`: Set to True to show bonus visualizations (default: False)

### Examples:

1. Train the model and show bonus visualizations:
python main.py data/car_data.csv --train True --bonus True
Copy
2. Predict price for a specific mileage using a trained model:
python main.py data/car_data.csv --mileage 50000
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
