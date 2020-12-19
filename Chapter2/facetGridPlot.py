# Import the required libraries
import matplotlib.pyplot as plt
import seaborn as sns
# Load the Iris dataset
iris = sns.load_dataset("iris")
# Generate the Facet Grid plot
sns.FacetGrid(iris,hue="species",height=6) \
.map(plt.scatter,"sepal_length","sepal_width") \
.add_legend()
# Display the plot
plt.show()
