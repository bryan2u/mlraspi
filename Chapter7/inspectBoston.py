# Load the required libraries
import pandas as pd
from sklearn.datasets import load_boston
# Load the Boston housing dataset
boston_dataset = load_boston()
# Display the dataset keys
print(boston_dataset.keys())
# Display the first five records
boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
print(boston.head())
# Display the extensive dataset description key
print(boston_dataset.DESCR)
