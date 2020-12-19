# Import required libraries
import matplotlib.pyplot as plt
import pandas as pd
def load_data(path, header):
    marks_df = pd.read_csv(path, header=header)
    return marks_df
if __name__ == "__main__":
    # load the data from the file
    data = load_data("marks.txt", None)
    # X = feature values, all the columns except the last column
    X = data.iloc[:, :-1]
    # y = target values, last column of the data frame
    y = data.iloc[:, -1]
    # Filter the applicants admitted
    admitted = data.loc[y == 1]
    # Filter the applicants not admitted
    not_admitted = data.loc[y == 0]
    # Display the dataset plot
    plt.scatter(admitted.iloc[:, 0], admitted.iloc[:, 1], s=10, label="Admitted")
    plt.scatter(not_admitted.iloc[:, 0], not_admitted.iloc[:, 1], s=10, label='Not Admitted')
    plt.legend()
    plt.show() 
