#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 15:15:11 2018

@author: joseph
"""

import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import sklearn
from pandas import Series, DataFrame
from pylab import rcParams
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn import metrics 
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


rcParams['figure.figsize'] = 10, 8
sb.set_style('whitegrid')

train_data = pd.read_csv('train_1stTeam.csv')
"""
train_data.drop(['Pos'],axis=1,inplace=True)

X = train_data.ix[:,(0,1,2,4)].values
y = train_data.ix[:,3].values
"""
X = train_data.ix[:,(0,1,2,3,5)].values
y = train_data.ix[:,4].values
"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state=0)
LogReg = LogisticRegression()
LogReg.fit(X_train, y_train)

X_new = pd.read_csv('test.csv')   
X_new = X_new[['BPM','PER','Pos','Pts','WS']]
y_pred = LogReg.predict_proba(X_new)
print(y_pred)

"""
y_pred = LogReg.predict(X_test)


#y_pred = LogReg.predict_proba(X_test)

print(y_pred)

confusion_matrix = confusion_matrix(y_test, y_pred)

print(confusion_matrix)

print(classification_report(y_test, y_pred))
