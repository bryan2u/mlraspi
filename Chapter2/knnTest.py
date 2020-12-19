# Import required libraries
import pandas
from kNN import kNN
from sklearn.metrics import mean_squared_error
# Read the training CSV file
training_data = pandas.read_csv("auto_train.csv")
x = training_data.iloc[:,:-1]
y = training_data.iloc[:,-1]
# Read the test CSV file
test_data = pandas.read_csv("auto_test.csv")
x_test = test_data.iloc[:,:-1]
y_test = test_data.iloc[:,-1]
# Display the heads from each CSV file
print('Training data')
print(training_data.head())
print('Test data')
print(test_data.head())
# Compute errors for k = 1, 3, and 20 with no weighting
for k in [1, 3, 20]:
    classifier = kNN(x,y,k)
    pred_test = classifier.predict(x_test)
    test_error = mean_squared_error(y_test, pred_test)
    print('Test error with k={}: {}'.format(k, test_error * len(y_test)/2))
# Compute errors for k = 1, 3, and 20 with weighting
for k in [1, 3, 20]:
    classifier = kNN(x,y,k,weighted=True)
    pred_test = classifier.predict(x_test)
    test_error = mean_squared_error(y_test, pred_test)
    print('Test error with k={}: {}'.format(k, test_error * len(y_test)/2))
