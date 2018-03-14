# Mining_Web

參與人員：徐子崴、徐躍華、王瀚

此為一個預測分析 NBA Data 的平常，主要有三個功能
功能：
1.名人堂預測：該選中球員如果當下退役，進入名人堂的機率為何？
  Data來源：https://www.basketball-reference.com/players
  前處理：由於三分球的起始年代約於1980年代左右，因此，在這之前的Data缺漏太多，即使做了資料補齊，誤差依然會很大。
         因此我將他全部清除了。
  算法：因為在那之前的名人堂球員實在是非常少數，我的training data為已退役球員的資料，
       我將三個csv檔的資料作交叉比對，取出了退役跟現役球員的資料，又將退役球員的label分為有進名人堂跟沒進名人堂，
       發現了有dataimbalance的問題，有進名人堂的資料約佔10%左右，我就把有進名人堂的資料放大了8倍，讓Data更平衡一些，
       再來我將traindata在拆成7:3，利用7份去訓練randomforest模型，再利用剩下的3份去測試準確度，測出來的準確度(oob_score)最高有到接近90%，
       最後再將現役球員的data丟進去最測試，看符不符合目前的各大運動網站上的預測，發現雖然順序有些不同，但是大致上會進球員都十分相似。

2.年度防守第一隊、年度第一隊：
  Data來源：https://www.basketball-reference.com/players
  前處理：由於得到的Data都是屬於原始資訊，我們將資料經過ＮＢＡ官網參考的公式計算後，算出進階的數據。
  算法：利用所得到的進階Data做Training，利用Random Forest 以及Linear Regression 做出Model之後，再去預測該選中球員的入選的機率。
  
Server概述：
1.Django Server:
  主要處理 Machine Learning 演算法，訓練出 Model 之後，利用 Express Server送過來的request data，去預測該球員的三種機率(名人堂、年度第一隊、防   守第一隊)，再將機率送回 Express Server。
2.Express Server:
  主要負責前端介面與送出使用者選中的球員數據，讓Django Server去做計算，然後再將 Return 回來的機率顯示在頁面上。
  
Demo 影片：https://youtu.be/4EQf-GIAVLY
