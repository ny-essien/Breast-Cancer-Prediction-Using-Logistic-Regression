# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 20:03:16 2023

@author: nsika
"""

import numpy as np
import pandas as pd

path = "c://ml_dataset/breast_cancer.csv"

"## Importing the dataset"
data = pd.read_csv(path)

"## Creating the independent varible X and the dependent y"
X = data.iloc[:, 1:-1].values
y = data.iloc[:,-1].values

"## Spliting into training set and test set"
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= 0.2, random_state= 42)

"## Training the logistic regression model on the training set"
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=42)
classifier.fit(X_train, y_train)

"## making prediction on the test set"
y_pred =  classifier.predict(X_test)

"## Making the confusion matrix"
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)

accuracy = accuracy_score(y_test, y_pred)
#printing the confusion matrix and the accuracy
print(f"Confusion Matrix : {cm}\nAccuracy : {accuracy}")
