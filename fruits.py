import numpy as np ##importing libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import sys

np.set_printoptions(precision=2)


fruits = pd.read_table('/home/piyush/Desktop/course3_downloads/fruit_data_with_colors.txt') ##path to the dataset

feature_names_fruits = ['height', 'width', 'mass', 'color_score'] ##feature of the fruits used for thr classification
X_fruits = fruits[feature_names_fruits]
y_fruits = fruits['fruit_label']
target_names_fruits = ['apple', 'mandarin', 'orange', 'lemon'] ##labels

X_fruits_2d = fruits[['height', 'width']]
y_fruits_2d = fruits['fruit_label']

X_train, X_test, y_train, y_test = train_test_split(X_fruits, y_fruits, random_state=0) 

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)## scaling of the training data 
# we must apply the scaling to the test set that we computed for the training set
X_test_scaled = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors = 5) ## k value of the knn set as 5 (hyper parameter)
knn.fit(X_train_scaled, y_train)
print('Accuracy of K-NN classifier on training set: {:.2f}' ##accuracy on training set
     .format(knn.score(X_train_scaled, y_train)))
print('Accuracy of K-NN classifier on test set: {:.2f}' ##accuracy on test set
     .format(knn.score(X_test_scaled, y_test)))
example_fruit = [[sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4]]] ##predicting the fruit by giving the required feature values from the terminal
example_fruit_scaled = scaler.transform(example_fruit) ## scaling the given values from the user
print('Predicted fruit type for ', example_fruit, ' is ', 
          target_names_fruits[knn.predict(example_fruit_scaled)[0]-1]) ##final prediction 

