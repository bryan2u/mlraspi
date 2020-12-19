# Import the required libraries
import matplotlib.pyplot as plt
import seaborn as sns
# Load the Iris dataset
iris = sns.load_dataset("iris")
# Generate the box plot
sns.boxplot(x="species",y="sepal_length", data=iris)
# Display the plot
plt.show()
