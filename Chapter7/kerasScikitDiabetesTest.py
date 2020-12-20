# Load required libraries
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import pandas as pd
# Function to create model, required for the KerasClassifier
def create_model():
    # create model
    model = Sequential()
    model.add(Dense(12, input_dim=8, activation="relu"))
    model.add(Dense(8, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])
    return model
# fix random seed for reproducibility
seed = 42
# Load the dataset
data = pd.read_csv('diabetes.csv')
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
# Split the dataset into 80% training and 20% testing sets   
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
# Impute the missing values using feature median values
imputer = SimpleImputer(missing_values=0,strategy='median')
X_train2 = imputer.fit_transform(X_train)
X_test2 = imputer.transform(X_test)
# Convert the numpy array into a Dataframe
X_train3 = pd.DataFrame(X_train2)
# create model
model = KerasClassifier(build_fn=create_model, epochs=150, batch_size=10, verbose=0)
# evaluate using 10-fold cross validation
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
# Evaluate using cross_val_score function
results = cross_val_score(model, X_train2, y_train, cv=kfold)
print(results.mean())
