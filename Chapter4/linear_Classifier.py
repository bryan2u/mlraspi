# Import the required libraries
import numpy as np
import cv2
# Initialize class labels and set pseudo-random seed value
labels = ['dog', 'cat', 'squirrel']
np.random.seed(1)
# Randomly initialize the weighting matrix and bias vector
W = np.random.randn(3, 3072)
b = np.random.randn(3)
# Set the font used to draw the label
font = cv2.FONT_HERSHEY_SIMPLEX
# Load the image and resize it. The image is taken from the dataset.
orig = cv2.imread('dog.png')
image = cv2.resize(orig, (32,32)).flatten()
# Compute output scores
scores = W.dot(image) + b
# Loop over the scores and labels
for (label, score) in zip(labels, scores):
   print('[INFO] {}: {:.2f}'.format(label, score))
# Get the class label for the highest scoring class
classLabel = labels[np.argmax(scores)]
# Draw the predicted label on the original image
cv2.putText(orig, classLabel, (10,30),  font, 0.9, (255,0,0), 2)
# Display image
cv2.imshow('Image', orig)
cv2.waitKey(0)
