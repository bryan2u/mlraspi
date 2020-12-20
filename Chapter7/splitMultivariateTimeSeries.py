# Import required libraries
from numpy import array
from numpy import hstack
# Split a multivariate sequence into samples
def split_sequences(sequences, n_steps):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)
# Define input sequences
in_seq1 = array([50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650])
in_seq2 = array([50,  75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350])
out_seq = array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])
# Convert to [rows, columns] structure
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
out_seq = out_seq.reshape((len(out_seq), 1))
# Horizontally stack columns
dataset = hstack((in_seq1, in_seq2, out_seq))
# Choose a number of time steps
n_steps = 3
# Convert into input/output samples
X, y = split_sequences(dataset, n_steps)
print(X.shape, y.shape)
# Display the data
for i in range(len(X)):
    print(X[i], y[i])
