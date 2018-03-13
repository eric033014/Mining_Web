#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 00:14:12 2018

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
from sklearn.cross_validation import train_test_split
from sklearn import metrics 
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


rcParams['figure.figsize'] = 10, 8
sb.set_style('whitegrid')

train = pd.read_csv('defence.csv')
train['STLBLK%'] = train['STL%'] + train['BLK%']

train.drop(['BLK%','STL%','GP'],axis=1,inplace=True)
print(train.head())


X = train.ix[:,(0,1,2,4)].values
y = train.ix[:,3].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state=0)
LogReg = LogisticRegression()
LogReg.fit(X_train, y_train)

"""
X_new = pd.read_csv('test.csv')   
X_new = X_new[['BPM','PER','Pts','WS']]
y_pred = LogReg.predict_proba(X_new)

print(y_pred)
"""


y_pred = LogReg.predict(X_test)

print(y_pred)


confusion_matrix = confusion_matrix(y_test, y_pred)

print(confusion_matrix)

print(classification_report(y_test, y_pred))
