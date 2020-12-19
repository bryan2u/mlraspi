# Import the required libraries
import matplotlib.pyplot as plt
import seaborn as sns
# Load the Iris dataset
iris = sns.load_dataset("iris")
# Generate the kde plot
sns.FacetGrid(iris,hue="species",height=6) \
.map(sns.kdeplot,"sepal_length") \
.add_legend()
# Display the plot
plt.show()
