# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.optimize as so
def load_data(path, header):
    # Load the CSV file into a panda dataframe
    marks_df = pd.read_csv(path, header=header)
    return marks_df
def sigmoid(x):
    # Activation function
    return 1/(1 + np.exp(-x))
def net_input(theta, x):
    # Computes the weighted sum of inputs by a numpy dot product
    return np.dot(x, theta)
def probability(theta, x):
    # Returns the probability after Sigmoid function is applied
    return sigmoid(net_input(theta, x))
def cost_function(theta, x, y):
    # Computes the cost function
    m = x.shape[0]
    total_cost = -(1/m)*np.sum(y*np.log(probability(theta,x))+(1-y)*np.log(1-probability(theta,x)))
    return total_cost
def gradient(theta, x, y):
    #Computes the cost function gradient
    m = x.shape[0]
    return (1/m)*np.dot(x.T,sigmoid(net_input(theta,x))-y)
def fit(x, y, theta):
    # The optimal coefficients are computed here
    opt_weights = so.fmin_tnc(func=cost_function, x0=theta, fprime=gradient,args=(x,y.flatten()))
    return opt_weights[0]
if __name__ == "__main__":
    # Load the data from the file
    data = load_data("marks.txt", None)
    # X = feature values, all the columns except the last column
    X = data.iloc[:, :-1]
    # Save a copy for the output plot
    X0 = X
    # y = target values, last column of the data frame
    y = data.iloc[:, -1]
    # Save a copy for the output plot
    y0 = y
    X = np.c_[np.ones((X.shape[0], 1)), X]
    y = y[:, np.newaxis]
    theta = np.zeros((X.shape[1], 1))
    parameters = fit(X, y, theta)
    x_values = [np.min(X[:,1]-5), np.max(X[:,2] + 5)]
    y_values = -(parameters[0] + np.dot(parameters[1], x_values)) / parameters[2]
    # filter the admitted applicants
    admitted = data.loc[y0 == 1]
    # filter the non-admitted applicants
    not_admitted = data.loc[y0 == 0]
    # Plot the original dataset along with the classifier line
    ax = plt.scatter(admitted.iloc[:, 0], admitted.iloc[:, 1], s=10, label="Admitted")
    ax = plt.scatter(not_admitted.iloc[:, 0], not_admitted.iloc[:, 1], s=10, label='Not Admitted')
    ax = plt.plot(x_values, y_values, label='Decision Boundary')
    ax = plt.xlabel('Marks in 1st Exam')
    ax = plt.ylabel('Marks in 2nd Exam')
    ax = plt.legend()
    plt.show()
    print(parameters)
