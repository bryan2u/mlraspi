# Import the required libraries
import matplotlib.pyplot as plt
import seaborn as sns
# Load the Iris dataset
iris = sns.load_dataset("iris")
# Generate the violin plot
sns.violinplot(x="species",y="sepal_length", data=iris, size=6)
# Display the plot
plt.show()
