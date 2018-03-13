import numpy as np
import pandas as pd
import re as re
import csv
import pip

train = pd.read_csv('Seasons_Stats.csv',header = 0,dtype = {'WS':np.float64,'MP':np.float64,'Year':np.float64,'TRB':np.float64,'AST':np.float64,'STL':np.float64,'BLK':np.float64,'PTS':np.float64})
now_player= pd.read_csv('now_player.csv',header=0)
fileHeader=["playername","TRB","AST","STL","BLK","PTS","fame","now","WS","MP"]
csv1=open("new_data.csv","w+")
csv2=open("old_data.csv","w+")
writer=csv.writer(csv1)
writer2=csv.writer(csv2)
writer.writerow(fileHeader)
writer2.writerow(fileHeader)
checklist=[]
trains=train
i=-1;
adjust=0
for player in trains['Player']:
    i=i+1
    print(player)
    if trains['Year'][i]<1974.0:
        continue
    #print(name)
    if player  not in checklist:
        if str(player)=="nan":
            continue
        tempdata=['',0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
        for check in now_player['player']:
            print("XXXXXXXXXXXXXXXXXXXXXXX "+str(check))
            print("XXXXXXXXXXXXXXXXXXXXXXX "+str(player))
            if str(check) in str(player):
                tempdata[7]=1
                break
        checklist.append(player)
        #tempdata=['',0,0,0,0,0,0]
        j=-1
        tempdata[0]=player
        if '*' in str(player):
            tempdata[6]=1
        else:
            tempdata[6]=0
        times=0
        for find in trains['Player']:
            j=j+1
            #if trains['Year'][j]>=1974.0:
            if player == find and trains['Year'][j]>=1974.0:
                times=times+1
                print ("same")
                tempdata[1]+=(trains['TRB'][j])
                tempdata[2]+=(trains['AST'][j])
                tempdata[3]+=(trains['STL'][j])
                tempdata[4]+=(trains['BLK'][j])
                tempdata[5]+=(trains['PTS'][j])
                tempdata[8]+=(trains['WS'][j])
                tempdata[9]+=(trains['MP'][j])
        tempdata[8]=tempdata[8]/times
        if tempdata[7]==1:
            writer.writerow(tempdata)
        else:
            if tempdata[6]==1:
                writer2.writerow(tempdata)
                writer2.writerow(tempdata)
                writer2.writerow(tempdata)
                writer2.writerow(tempdata)
                writer2.writerow(tempdata)
                writer2.writerow(tempdata)
       
            else:
                writer2.writerow(tempdata)
csv1.close()
csv2.close()
