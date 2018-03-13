from django.shortcuts import render
from django.shortcuts import render,render_to_response
import numpy as np
import csv
import pandas as pd
from sklearn import cross_validation, ensemble, preprocessing, metrics
import seaborn as sb
import matplotlib.pyplot as plt
import sklearn
from pandas import Series, DataFrame
from pylab import rcParams
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from django.views.decorators.csrf import csrf_exempt
import json
from django.http import StreamingHttpResponse
 






# Create your views here.
def halloffame(testdata):
    test = pd.read_csv('Web/new_data.csv',header = 0,dtype = {'WS':np.float64,'MP':np.float64,'TRB':np.float64,'AST':np.float64,'STL':np.float64,'BLK':np.float64,'PTS':np.float64})
    train = pd.read_csv('Web/old_data.csv',header = 0,dtype = {'WS':np.float64,'MP':np.float64,'TRB':np.float64,'AST':np.float64,'STL':np.float64,'BLK':np.float64,'PTS':np.float64})
    fileHeader=["playername","TRB","AST","STL","BLK","PTS","fame","now"]
    Train_x=pd.DataFrame([train['WS'],train['MP'],train['AST'],train['BLK'],train['PTS'],train['TRB'],train['STL']]).T
    Train_y=train['fame']
    Test_x=pd.DataFrame([test['WS'],test['MP'],test['AST'],test['BLK'],test['PTS'],test['TRB'],test['STL']]).T
    train_X, test_X, train_y, test_y = cross_validation.train_test_split(Train_x, Train_y, test_size = 0.3)
    forest = ensemble.RandomForestClassifier(n_estimators = 100,oob_score=True)
    forest_fit = forest.fit(train_X, train_y)
    #print("%.4f"%forest.oob_score_)
    # 預測
    test_y_predicted = forest.predict_proba(test_X)

    #print(train_y)
    print("-----------------------------------------")
    # 績效
    #accuracy = metrics.accuracy_score(test_y, test_y_predicted)

    answer=forest.predict_proba(testdata)[:,1]
    print(answer)
    return answer[0]*100
    #print(accuracy)


def firstdefense(testdata):
    rcParams['figure.figsize'] = 10, 8
    sb.set_style('whitegrid')

    train = pd.read_csv('Web/defence.csv')
    train['STLBLK%'] = train['STL%'] + train['BLK%']

    train.drop(['BLK%','STL%','GP'],axis=1,inplace=True)
    print(train.head())


    X = train.ix[:,(0,1,2,4)].values
    y = train.ix[:,3].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state=0)
    LogReg = LogisticRegression()
    LogReg.fit(X_train, y_train)


    #X_new = pd.read_csv('test.csv') 
    X_new = testdata
    #X_new = X_new[['DBPM','DWS','Pos','STL%','BLK%']]
    X_new['STLBLK%'] = X_new['STL%'] + X_new['BLK%']
    X_new.drop(['BLK%','STL%'],axis=1,inplace=True)
    y_pred = LogReg.predict_proba(X_new)

    print(y_pred)
    return y_pred[:,1][0]*100

def firstteam(testdata):
    rcParams['figure.figsize'] = 10, 8
    sb.set_style('whitegrid')

    train_data = pd.read_csv('Web/train_1stTeam.csv')
    X = train_data.ix[:,(0,1,2,3,5)].values
    y = train_data.ix[:,4].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state=0)
    LogReg = LogisticRegression()
    LogReg.fit(X_train, y_train)

    X_new = testdata   
    #X_new = X_new[['BPM','PER','Pos','Pts','WS']]
    y_pred = LogReg.predict_proba(X_new)
    print(y_pred)
    return y_pred[:,1][0]*100

@csrf_exempt
def home(request):
    received_json_data=json.loads(request.body)
    print(type(received_json_data['WS']))
    data1=pd.DataFrame([float(received_json_data['ws']),float(received_json_data['MP']),float(received_json_data['AST']),float(received_json_data['BLK']),float(received_json_data['PTS']),float(received_json_data['TRB']),float(received_json_data['STL'])]).T
    #Test_x=pd.DataFrame([test['WS'],test['MP'],test['AST'],test['BLK'],test['PTS'],test['TRB'],test['STL']]).T
    if received_json_data['Pos']=='C':
        pos=2
    elif received_json_data['Pos']=='SF' or received_json_data['Pos']=='PF':
        pos=1
    else:
        pos=0

    data2=pd.DataFrame({'DBPM':[float(received_json_data['DBPM'])],'DWS':[float(received_json_data['DWS'])],'Pos':[pos],'STL%':[float(received_json_data['STLp'])],'BLK%':[float(received_json_data['BLKp'])]})
    #X_new = X_new[['BPM','PER','Pts','WS']]
 
    data3=pd.DataFrame({'BPM':[float(received_json_data['BPM'])],'PER':[float(received_json_data['PER'])],'Pos':[pos],'Pts':[float(received_json_data['Pts'])],'WS':[float(received_json_data['WS'])]})
    #X_new = X_new[['BPM','PER','Pos','Pts','WS']]
    answer1=halloffame(data1)
    answer2=firstdefense(data2)
    answer3=firstteam(data3)
    return StreamingHttpResponse(str(answer1)+":"+str(answer2)+":"+str(answer3))
    return render_to_response('home.html',locals() )

@csrf_exempt
def result(request):
    if request.method=="POST":
        #received_json_data=json.loads(request.POST['data'])
        received_json_data=json.loads(request.body)
        return StreamingHttpResponse('it was post request: '+str(received_json_data))
        print ("POST!!!!!!!")
        print(request.POST)
    else:
        print(request.GET)
    return render_to_response('result.html',locals())
