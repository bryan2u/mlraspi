# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
# Load the dataset
data = pd.read_csv('diabetes.csv')
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
# Split the dataset into 80% training and 20% testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
# Impute the missing values using feature median values
imputer = SimpleImputer(missing_values=0, strategy="median")
X_train2 = imputer.fit_transform(X_train)
X_test2 = imputer.transform(X_test)
# Convert the numpy array into a Dataframe
X_train3 = pd.DataFrame(X_train2)
# Display the first 10 records
print(X_train3.head(10))
def plotHistogram(values, label, feature, title):
    sns.set_style("whitegrid")
    plotOne = sns.FacetGrid(values, hue=label, aspect=2)
    plotOne.map(sns.distplot, feature, kde=False)
    plotOne.set(xlim=(0, values[feature].max()))
    plotOne.add_legend()
    plotOne.set_axis_labels(feature, 'Proportion')
    plotOne.fig.suptitle(title)
    plt.show()
# Plot the heathy patient histograms for insulin and skin
# thickness
plotHistogram(X_train3,None,4,'Insulin vs Diagnosis')
plotHistogram(X_train3,None,3,'SkinThickness vs Diagnosis')
# Check to see if any 0's remain
data2 = X_train2
print("Num of Rows, Num of Columns: ", data2.shape)
print("\nColumn Name          Num of Null Values\n")
print((data2[:] == 0).sum())
print("Num of Rows, Num of Columns: ", data2.shape)
print("\nColumn Name          %Null Values\n")
print(((data2[:] == 0).sum()) / 614 * 100)
# Display the correlation matrix
corr1 = X_train3.corr()
print(corr1)
