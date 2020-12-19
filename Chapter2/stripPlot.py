# Import the required libraries
import matplotlib.pyplot as plt
import seaborn as sns
# Load the Iris dataset
iris = sns.load_dataset("iris")
# Generate the strip plot
ax = sns.boxplot(x="species",y="sepal_length", data=iris)
ax = sns.stripplot(x="species", y="sepal_length", data=iris, jitter=True, edgecolor="gray")
# Display the plot
plt.show()
