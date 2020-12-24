# Import required libraries libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
# Load data
data = pd.read_csv('input/data.csv')
#del data['Unnamed: 32']
X = data.iloc[:, 1:9].values
y = data.iloc[:, 10].values
# Encoding categorical data
labelencoder_X_1 = LabelEncoder()
y = labelencoder_X_1.fit_transform(y)
# Split the dataset into Training and Test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)
#Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# Initialise the ANN
classifier = Sequential()
# Add the input layer and the first hidden layer
classifier.add(Dense(units=16, kernel_initializer='uniform', activation='relu', input_dim=8))
# Add dropout to prevent overfitting
classifier.add(Dropout(0.1))
# Add the second hidden layer
classifier.add(Dense(units=16, kernel_initializer='uniform', activation='relu'))
# Add dropout to prevent overfitting
classifier.add(Dropout(0.1))
# Add the output layer
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
# Compile the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# Fit the ANN to the Training set
# The batch size and number of epochs have been set using trial
# and error.
classifier.fit(X_train, y_train, batch_size=100, epochs=150)
# Predict the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5) # Converts continuous to binary
#  Create confusion matrix object
cm = confusion_matrix(y_test, y_pred)
# Display accuracy
print('Accuracy is {}%'.format(((cm[0][0] + cm[1][1])/70)*100))
# Display the confusion matrix
print('\nConfusion Matrix\n',cm)
# Generate and display a Seaborn heatmap
sns.heatmap(cm, annot=True)
plt.savefig('bcHeatmap.png')
plt.show()
# Instantiate a random forest classifier
rf_clf = RandomForestClassifier(n_estimators=100)
rf_clf.fit(X_train, y_train)
# Compute the probability distributions
probas = rf_clf.predict_proba(X_test)# plot
plt.figure(dpi=150)
plt.hist(probas, bins=20)
plt.title('Classification Probabilities')
plt.xlabel('Probability')
plt.ylabel('# of Instances')
plt.xlim([0.5, 1.0])
plt.legend('01')
plt.show()
# Compute the false and true positive rates
fpr, tpr, thresholds = roc_curve(y_test, probas[:,0], pos_label=0)
# Compute the area under the curve
roc_auc = auc(fpr, tpr)
# Plot the AUROC curve
plt.figure(dpi=150)
plt.plot(fpr, tpr, lw=1, color='green', label=f'AUC = {roc_auc:.3f}')
plt.title('ROC Curve for RF classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.legend()
plt.show()
