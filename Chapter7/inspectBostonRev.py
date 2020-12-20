# Load the required libraries
import pandas as pd
from sklearn.datasets import load_boston
# Load the Boston housing dataset
boston_dataset = load_boston()
# Display the dataset keys
print(boston_dataset.keys())
# Create the boston Dataframe
boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
# Add the target variable to the Dataframe
boston['MEDV'] = boston_dataset.target
# Display the first five records
print(boston.head())
# Check for null values in the dataset
print(boston.isnull().sum())
