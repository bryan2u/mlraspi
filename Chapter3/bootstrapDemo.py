# Import required libraries
from random import seed
from random import random
from random import randrange
# Create a random sub-set from the dataset with replacement.
def subsample(dataset, ratio=1.0):
      sample = list()
      n_sample = round(len(dataset) * ratio)
      while len(sample) < n_sample:
            index = randrange(len(dataset))
            sample.append(dataset[index])
      return sample
 # Calculate the mean of a list of numbers.
def mean(numbers):
      return sum(numbers) / float(len(numbers))
seed(1)
# Calculate the true mean.
# The original dataset has 20 rows with a single random
# number 0 to 9 in each row.
dataset = [[randrange(10)] for i in range(20)]
print('True Mean: %.3f' % mean([row[0] for row in dataset]))
# Calculate and display the estimated means from the different
# sized sub-sets.
ratio = 0.10
for size in [1, 10, 100]:
      sample_means = list()
      for i in range(size):
            sample = subsample(dataset, ratio)
            sample_mean = mean([row[0] for row in sample])
            sample_means.append(sample_mean)
      print('Samples=%d, Estimated Mean: %.3f' % (size, mean(sample_means)))
