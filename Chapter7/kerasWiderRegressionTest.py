# Import required libraries
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_boston
# Load the Boston housing dataset
boston_dataset = load_boston()
# Create the boston Dataframe
dataframe = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
#  Add the target variable to the dataframe
dataframe['MEDV'] = boston_dataset.target
# Setup the boston dataframe
boston = dataframe.values
# Split into input (X) and output (y) variables
X = boston[:,0:13]
y = boston[:,13]
# Define the wider model
def wider_model():
    # create model
    model = Sequential()
    model.add(Dense(20, input_dim=13, kernel_initializer="normal", activation="relu"))
    model.add(Dense(1, kernel_initializer="normal"))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer="adam")
    return model
# Random seed for reproducibility
seed = 42
# Create a regression object
estimator = KerasRegressor(build_fn=wider_model, epochs=100, batch_size=5, verbose=0)
# Evaluate model with standardized dataset
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=wider_model, epochs=50, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(pipeline, X, y, cv=kfold)
print("Wider: %.2f (%.2f) MSE" % (results.mean(), results.std()))
