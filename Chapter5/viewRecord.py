# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
# Create data list
dataFile = open('mnist_train_100.csv')
dataList = dataFile.readlines()
dataFile.close()
# Get the record number
print('Enter record number to be viewed: ', end = ' ')
num = input()
# Get the record
record = dataList[int(num)].split(',')
# Reshape the array for imaging
imageArray = np.asfarray(record[1:]).reshape(28,28)
# Image it as a grayscale
plt.imshow(imageArray, cmap="Greys", interpolation="None")
plt.show()
