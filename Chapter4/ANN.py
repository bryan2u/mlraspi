# Import required libraries
import numpy as np
class ANN:
    def __init__(self, inode, hnode, onode, lr):
        # Set local variables
        self.inode = inode
        self.hnode = hnode
        self.onode = onode
        self.lr = lr
        # Mean is the reciprocal of the sqrt of total nodes
        mean = 1/(pow((inode + hnode + onode), 0.5))
        # Std dev is approx 1/6 of total range
        # Range = 2
        sd = 2/6
        # Generate both weight matrices
        # Input to hidden layer matrix
        self.wtgih = np.random.normal(mean, sd, [hnode, inode])
        # Hidden to output layer matrix
        self.wtgho = np.random.normal(mean, sd, [onode, hnode])
    def testNet(self, input):
        # Convert input data vector into numpy array
        input = np.array(input, ndmin=2).T
        # Multiply input by wtgih
        hInput = np.dot(self.wtgih, input)
        # Apply activation function
        hOutput = 1/(1 + np.exp(-hInput))
        # Multiply hidden layer output by wtgho
        oInput = np.dot(self.wtgho, hOutput)
        # Apply activation function
        oOutput = 1/(1 + np.exp(-oInput))
        return oOutput
    def trainNet(self, inputT, train):
        # This module depends upon values, arrays and matrices
        # created when the init module is run
        # Create the arrays from the arguments
        self.inputT = np.array(inputT, ndmin=2).T
        self.train = np.array(train, ndmin=2).T
        # Multiply inputT array by wtgih
        self.hInputT = np.dot(self.wtgih, self.inputT)
        # Apply activation function
        self.hOutputT = 1/(1 + np.exp(-self.hInputT))
        # Multiply hidden layer output by wtgho
        self.oInputT = np.dot(self.wtgho, self.hOutputT)
        # Apply activation function
        self.oOutputT = 1/(1 + np.exp(-self.oInputT))
        # Calculate output errors
        self.eOutput = self.train - self.oOutputT
        # Calculate hidden layer error array
        self.hError = np.dot(self.wtgho.T, self.eOutput)
        # Update weight matrix wtgho
        self.wtgho += self.lr*np.dot((self.eOutput*self.oOutputT*(1 - self.oOutputT)), self.hOutputT.T)
        # Update weight matrix wtgih
        self.wtgih += self.lr*np.dot((self.hError*self.hOutputT*(1 - self.hOutputT)), self.inputT.T)
