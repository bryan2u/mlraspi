# Import required libraries
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
# Load the CSV dataset
dataset = pd.read_csv('diabetes.csv')
dataset.head(10)
# Define a histogram plot method
def plotHistogram(values, label, feature, title):
    sns.set_style("whitegrid")
    plotOne = sns.FacetGrid(values, hue=label, aspect=2)
    plotOne.map(sns.distplot, feature, kde=False)
    plotOne.set(xlim=(0, values[feature].max()))
    plotOne.add_legend()
    plotOne.set_axis_labels(feature, 'Proportion')
    plotOne.fig.suptitle(title)
    plt.show()
# Plot the Insulin histogram
plotHistogram(dataset, 'Outcome', 'Insulin', 'Insulin vs Diagnosis (Blue = Healthy; Orange = Diabetes)')
# Plot the SkinThickness histogram
plotHistogram(dataset, 'Outcome', 'SkinThickness', 'SkinThickness vs Diagnosis (Blue = Healthy; Orange = Diabetes)')
# Summary of the number of 0's present in the dataset by feature
dataset2 = dataset.iloc[:, :-1]
print("Num of Rows, Num of Columns: ", dataset2.shape)
print("\nColumn Name          Num of Null Values\n")
print((dataset[:] == 0).sum())
# Percentage summary of the number of 0's in the dataset
print("Num of Rows, Num of Columns: ", dataset2.shape)
print("\nColumn Name          %Null Values\n")
print(((dataset2[:] == 0).sum()) / 768 * 100)
# Create a heat map
g = sns.heatmap(dataset.corr(), cmap="BrBG", annot=False)
plt.show()
# Display the feature correlation values
corr1 = dataset.corr()
print(corr1[:])
