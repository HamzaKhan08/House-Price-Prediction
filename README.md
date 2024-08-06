# House Price Prediction by Hamza Khan

This project predicts house prices using a machine learning model trained on the Boston Housing dataset. The model employs data preprocessing techniques and a Random Forest Regressor to forecast median house values (`MEDV`). The dataset is divided into training, validation, and test sets to optimize model performance.

## Dataset

The dataset used in this project is the [Boston Housing dataset](https://www.kaggle.com/c/boston-housing), which contains information collected by the U.S. Census Service concerning housing in the area of Boston, Massachusetts. The dataset includes the following columns:

- `CRIM`: per capita crime rate by town
- `ZN`: proportion of residential land zoned for lots over 25,000 sq. ft.
- `INDUS`: proportion of non-retail business acres per town
- `CHAS`: Charles River dummy variable (1 if tract bounds river; 0 otherwise)
- `NOX`: nitric oxides concentration (parts per 10 million)
- `RM`: average number of rooms per dwelling
- `AGE`: proportion of owner-occupied units built prior to 1940
- `DIS`: weighted distances to five Boston employment centres
- `RAD`: index of accessibility to radial highways
- `TAX`: full-value property-tax rate per $10,000
- `PTRATIO`: pupil-teacher ratio by town
- `B`: 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
- `LSTAT`: % lower status of the population
- `SalePrice`: Median value of owner-occupied homes in $1000s

## Installation

To run this project, follow these steps:

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/house-price-prediction.git
   cd house-price-prediction
2. **Set Up a Virtual Environment

It is recommended to use a virtual environment to manage dependencies. You can create one using venv:
python3 -m venv .venv
source .venv/bin/activate  # On Windows, use .venv\Scripts\activate

3. **Install Dependencies

Install the required packages using pip:
pip install -r requirements.txt

# Usage
Prepare the Dataset

Ensure the dataset file data.csv is in the project directory. This file should contain the Boston Housing dataset.

Run the Code

Execute the Python script to preprocess the data, train the model, and make predictions:
python house_price_prediction.py

# Outputs

The script will generate a correlation matrix and visualize the distribution of SalePrice.
It will split the dataset into training, validation, and test sets.
The model's performance is evaluated using Mean Absolute Error (MAE) on the validation set.
Predictions are saved to submission.csv.

## Project Structure
house-price-prediction/
│
├── data.csv                       # Dataset file
├── house_price_prediction.py      # Main script for training and prediction
├── requirements.txt               # List of Python dependencies
└── README.md                      # Project documentation

# Dependencies
The project requires the following Python libraries:

pandas
numpy
seaborn
matplotlib
scikit-learn

# License
This project is licensed under the MIT License - see the LICENSE file for details.

# Acknowledgements
This project uses the Boston Housing dataset, which is available at Kaggle.


### Key Sections:

- **Project Overview**: Briefly describes the purpose and scope of the project.
- **Dataset**: Lists and explains the dataset features.
- **Installation**: Provides instructions on how to set up the project locally.
- **Usage**: Describes how to prepare the data and run the script.
- **Project Structure**: Outlines the files and directories in the project.
- **Dependencies**: Lists the required Python packages.
- **Contributing**: Encourages community involvement.
- **License**: Mentions the licensing for the project.
- **Acknowledgements**: Gives credit to data sources or collaborators.

Feel free to modify this README to better fit your project's specifics or personal preferences. Let me know if you need any additional changes or information!

