# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
from ANN import ANN
# Setup the ANN configuration
inode =784
hnode =100
onode =10
# Set the learning rate
lr = 0.2
# Instantiate an ANN object named ann
ann = ANN(inode, hnode, onode, lr)
# Create the training list data
dataFile = open('mnist_train_100.csv')
dataList = dataFile.readlines()
dataFile.close()
# Train the ANN using all the records in the list
for record in dataList:
      recordx = record.split(',')
      inputT = (np.asfarray(recordx[1:])/255.0*0.99) + 0.01
      train = np.zeros(onode) + 0.01
      train[int(recordx[0])] =0.99
      # Training begins here
      ann.trainNet(inputT, train)
