# Import required libraries
import numpy as np
# Create the input data vector
input = np.array([0.8, 0.2, 0.7])[:,None]
# Create the wtgih matrix
wtgih = np.matrix([[0.8, 0.6, 0.3], \
                   [0.2, 0.9, 0.3], \
                   [0.2, 0.5, 0.8]])
# Create the wtgho matrix
wtgho = np.matrix([[0.4, 0.8, 0.4], \
                   [0.5, 0.7, 0.2], \
                   [0.9, 0.1, 0.6]])
# Compute the dot product of the input vector and wtgih matrix
X1 = np.dot(input.T, wtgih)
# Display the matrix
print('X1 matrix\n', X1)
print()
# Apply the activation function to the X1 matrix
out1 = 1 / (1 + np.exp(-X1))
# Display the matrix
print('out1 matrix\n', out1)
print()
# Compute the dot product of the X1 and wtgho matrices
X2 = np.dot(out1, wtgho)
# Display the matrix
print('X2 matrix\n', X2)
print()
# Apply the activation function to the X2 matrix
out2 = 1 / (1 + np.exp(-X2))
# Display the matrix
print('out2 matrix\n', out2)
