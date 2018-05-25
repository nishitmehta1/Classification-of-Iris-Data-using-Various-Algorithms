# -*- coding: utf-8 -*-
"""
Created on Thu May 24 18:22:28 2018

@author: Nishit Mehta
"""
#%%
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
#%%
data = pd.read_csv('iris.csv')
data_test = pd.read_csv('iris_test.csv')

data = data.sample(frac=1).reset_index(drop=True)
data_test = data_test.sample(frac=1).reset_index(drop=True)
#%%
x = data.iloc[:,:-1].values
y = data.iloc[:,4].values

x_test = data_test.iloc[:,:-1].values
y_test = data_test.iloc[:,4].values
#%%
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x, y)
#%%
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(x, y)
#%%
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(x, y)
#%%
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(x, y)
#%%
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(x, y)
#%%
y_pred = classifier.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
accuracy = classifier.score(x_test, y_test)
y_pred = classifier.predict(x_test)
print (cm)
print ("Accuracy =", accuracy*100)
'''
CLASSIFICATION REPROT
'''
target_names = ['Iris-setosa', 'Iris-versicolor','Iris-virginica']
print(classification_report(y_test, y_pred, target_names= target_names))