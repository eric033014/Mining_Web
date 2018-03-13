import numpy as np
import csv
import pandas as pd
from sklearn import cross_validation, ensemble, preprocessing, metrics
test = pd.read_csv('new_data.csv',header = 0,dtype = {'WS':np.float64,'MP':np.float64,'TRB':np.float64,'AST':np.float64,'STL':np.float64,'BLK':np.float64,'PTS':np.float64})
train = pd.read_csv('old_data.csv',header = 0,dtype = {'WS':np.float64,'MP':np.float64,'TRB':np.float64,'AST':np.float64,'STL':np.float64,'BLK':np.float64,'PTS':np.float64})
fileHeader=["playername","TRB","AST","STL","BLK","PTS","fame","now"]
Train_x=pd.DataFrame([train['WS'],train['MP'],train['AST'],train['BLK'],train['PTS'],train['TRB'],train['STL']]).T
Train_y=train['fame']
Test_x=pd.DataFrame([test['WS'],test['MP'],test['AST'],test['BLK'],test['PTS'],test['TRB'],test['STL']]).T
train_X, test_X, train_y, test_y = cross_validation.train_test_split(Train_x, Train_y, test_size = 0.3)
forest = ensemble.RandomForestClassifier(n_estimators = 100,oob_score=True)
forest_fit = forest.fit(train_X, train_y)
print("%.4f"%forest.oob_score_)
# 預測
test_y_predicted = forest.predict_proba(test_X)

print(train_y)
print("-----------------------------------------")
# 績效
#accuracy = metrics.accuracy_score(test_y, test_y_predicted)

answer=forest.predict_proba(Test_x)[:,1]
print(answer)
#print(accuracy)


