# Import required libraries
from ANN import ANN
# Create input data vector
inputT = [0.8, 0.5, 0.6]
# Display it
print('Input data vector')
print(inputT)
print()
# Train for 1 iteration
train = inputT
ann = ANN(3,3,3,0.3)
output = ann.testNet(inputT)
# Display output
print('After one iteration')
print(output)
print()
# Train for 499 iterations
for i in range(499):
    ann.trainNet(inputT, train)
output = ann.testNet(inputT)
# Display output
print('After 500 iterations')
print(output)
print()
