# Multivariate data preparation
from numpy import array
from numpy import hstack
# Define input sequences
in_seq1 = array([50, 100, 150, 200, 250, 300, 350, 400, 450,
500, 550, 600, 650])
in_seq2 = array([50, 75, 100, 125, 150, 175, 200, 225, 250,
275, 300, 325, 350])
out_seq = array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])
# Convert to [rows, columns] structure
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
out_seq = out_seq.reshape((len(out_seq), 1))
# Horizontally stack columns
dataset = hstack((in_seq1, in_seq2, out_seq))
# Display the datasets
print(dataset)
