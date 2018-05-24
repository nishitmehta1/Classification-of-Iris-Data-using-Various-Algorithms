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
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
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
'''
KNN or NaiveBayes or SVM or Decision Tree or Random Forest
Which is better?
More than 2 Classes?
'''
classifier = GaussianNB()
classifier.fit(x, y)
#%%
y_pred = classifier.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
accuracy = classifier.score(x_test, y_test)
print (accuracy*100)