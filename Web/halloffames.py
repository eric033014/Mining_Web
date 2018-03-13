import numpy as np
import pandas as pd
import re as re
import requests  
from bs4 import BeautifulSoup
import csv
import pip
fileHeader=["player","totalgame","point","rebound","assist","fieldgoal","threepoint","freethrow","efg","per","ws"]
csv1=open("player.csv","w+")
writer=csv.writer(csv1)
writer.writerow(fileHeader)
character=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
for char in character:
    url="https://www.basketball-reference.com/players/"+char+"/"
    a=requests.get(url).text
    soup=BeautifulSoup(a,'lxml')
    player_url=soup.find_all('th',scope="row")

    for player in player_url:
        if(int(player.parent.find_all('td')[1].text)>=1981):
            print(player.find('a').text)
            name=player.find('a').text
            name_url="https://www.basketball-reference.com"+player.find('a')['href']
            b=requests.get(name_url).text
            soup1=BeautifulSoup(b,'lxml')
            data1=soup1.find('div',class_="stats_pullout").find('div',class_="p1")
            totalgame=data1.find_all('p')[1].text
            point=data1.find_all('p')[3].text
            rebound=data1.find_all('p')[5].text
            assist=data1.find_all('p')[7].text
            data2=soup1.find('div',class_="stats_pullout").find('div',class_="p2")
            fieldgoal=data2.find_all('p')[1].text
            threepoint=data2.find_all('p')[3].text
            freethrow=data2.find_all('p')[5].text
            efg=data2.find_all('p')[7].text
            data3=soup1.find('div',class_="stats_pullout").find('div',class_="p3")
            per=data3.find_all('p')[1].text
            ws=data3.find_all('p')[3].text
            arraylist=[name,totalgame,point,rebound,assist,fieldgoal,threepoint,freethrow,efg,per,ws]
            writer.writerow(arraylist)
csv1.close

#train = pd.read_csv('train.csv',header=0)

