# Usage
# python irisDemo.py
# Load libraries
import pandas 
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)
# Display the shape
print('Dataset dimensions')
print(dataset.shape)
# Display the first portion of the data
print('Head of the data')
print(dataset.head(20))
# Display data statistics
print('Statistics')
print(dataset.describe())
# Display class distribution
print('Class distribution')
print(dataset.groupby('class').size())
# Visualize data with box and whisker diagrams
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()
# Visualize data with histograms
dataset.hist()
plt.show()
# Visualize data with scatter plots
scatter_matrix(dataset)
plt.show()
