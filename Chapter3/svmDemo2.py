# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
# Import dataset into a variable named 'cancer'
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
# Load input features as a DataFrame
df_features = pd.DataFrame(cancer['data'], columns = cancer['feature_names'])
# Add output variable 'target' into a DataFrame
df_target = pd.DataFrame(cancer['target'], columns = ['Cancer'])
# Display the first 5 records
print(df_features.head())
# Split the dataset, 70% to train, 30% to test
X_train, X_test, y_train, y_test = train_test_split(df_features, np.ravel(df_target), test_size=0.30, random_state=101)
# Instantiate the SVC model. SVC is the sklearn classifier name.
model = SVC()
# Train the model using the fit method
model.fit(X_train, y_train)
# Test the model using the test dataset
predictions = model.predict(X_test)
# Display the prediction results
print(classification_report(y_test, predictions))

# Gridsearch
param_grid = {'C':[0.1, 1, 10, 100, 1000], 'gamma':[1, 0.1, 0.01, 0.001, 0.0001], 'kernel':['rbf']}
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)
grid.fit(X_train, y_train)
print('\n')
print('The best parameters are ', grid.best_params_)
grid_predictions = grid.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test, grid_predictions))
