import numpy as np
from VGG import VGG
from keras.datasets import fashion_mnist
from keras.utils import np_utils
from keras import backend as K
# K.set_image_dim_ordering('th')
K.set_image_data_format('channels_last')
# Set a random seed
seed = 42
np.random.seed(seed)
# Load the datasets
(train_images, train_labels),(test_images, test_labels) = fashion_mnist.load_data()
# Flatten all of the 28 x 28 images into 784 element numpy input
# data vectors.
pixelNum = train_images.shape[1] * train_images.shape[2]
train_images = train_images.reshape(train_images.shape[0],28,28,1).astype('float32')
test_images = test_images.reshape(test_images.shape[0],28,28,1).astype('float32')
# Normalize inputs from 0-255 to 0-1
train_images = train_images / 255.0
test_images = test_images / 255.0
# One hot encoding
train_labels = np_utils.to_categorical(train_labels)
test_labels = np_utils.to_categorical(test_labels)
numClass = test_labels.shape[1]
# Run the demo and evaluate it
vgg = VGG()
model = vgg.build(28, 28, 1, numClass)
train_images = train_images.reshape(60000, 28, 28, 1)
test_images = test_images.reshape(10000, 28, 28, 1)
model.fit(train_images, train_labels, validation_data=(test_images, test_labels), epochs=10, batch_size=100, verbose=2)
# Final evaluation
scores = model.evaluate(test_images, test_labels, verbose=0)
print(scores[1])
