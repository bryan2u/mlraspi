# Import the required libraries
import matplotlib.pyplot as plt
import seaborn as sns
# Load the Iris dataset
iris = sns.load_dataset("iris")
# Generate the pair plots
sns.pairplot(iris, hue="species", height=2.5)
# Display the plots
plt.show()
