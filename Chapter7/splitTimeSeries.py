# Import required library
from numpy import array
# Split a univariate time series into samples
def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)
# Define input time series
raw_seq = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650]
# Choose a number of time steps
n_steps = 3
# Split into samples
X, y = split_sequence(raw_seq, n_steps)
# Display the data
for i in range(len(X)):
    print(X[i], y[i])
