#!/usr/bin/env python
# coding: utf-8

# ## リファクタリングしてみる  
# 今回の目的としては、可読性が高まる？と勝手に思ってるが、関数に分けて処理を記述してみることにする。  
# この時、魔法みたいなmainのやつも使ってみることにした.

# In[5]:


import cv2
import numpy as np
from matplotlib import pyplot as plt
import glob
from natsort import natsorted
from statistics import mean

#ボーリング画像の取得. 
files = natsorted( glob.glob("IMAGE/TEST/LEVEL4_TEST_USE3/*") )

#立っているピンの数を格納する.
output = [] #これは正解を比較する時に配列同士の比較をしてるから記述.

#円検出処理.
for f in files:
    
    miss = 0 #1枚の画像でミスをした数
    
    #グレースケール読み込み
    img_gray = cv2.imread(f, 0)
    
    #バイラテラルフィルタより, HoughCirclesのパターメータを検出精度が高くなるように設定できた.
    #カーネルサイズは高い方が, ピン以外のエッジを検出しなくなる. 計算量も踏まえ１１とした.
    img_gray = cv2.medianBlur(img_gray, 11)  
    
    #円の印をつける画像をRGB表示するために読み込む.
    img_BGR = cv2.imread(f)
    cimg = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB) 
    
    #ハフ変換による円検出.
    circles = cv2.HoughCircles(
        img_gray,                      
        cv2.HOUGH_GRADIENT,
        1,                     
        50,                   
        param1=10,    #閾値を下げることで検出するエッジの数を増やした. 
        param2=30,    #高い方が誤検出を防げるので, 円をミスしない範囲で大きい値とした.
        minRadius=50, #円の半径の最大最小は事前に調べて範囲を絞った.
        maxRadius=60 
    )
    
    if circles is not None:
        #円が検出できた
            circles = np.uint16(np.around(circles))
            for i in circles[0,:]:
                pixelvalue = img_BGR[ i[1], i[0] ]
                if mean(pixelvalue) > 180 :
                    # 囲み線を描く
                    cv2.circle(cimg,(i[0], i[1]), i[2], (255, 0, 0), 10)
                    # 中心点を描く
                    cv2.circle(cimg,(i[0], i[1]), 2, (0,0,255), 3)
                else:
                    miss = miss + 1
            #出力を保存.
            hit_num = circles.shape[1] - miss #正解数
            output.append(hit_num)
            #print("立っているピンの数",hit_num, "本")
    else: 
        #円が検出できなかった.
        output.append(0)
        #print("立っているピンの数", "0", "本")
        
    #描画する. 
    #plt.imshow(cimg)
    #plt.show()


#正解データを読み込む.
Answer_data = open("ANS_TXT/LEVEL4_TEST_ANS.txt", "r")
lines = Answer_data.readlines()
Ans = []
for i in range (0,len(lines)):
    Ans.append(int(lines[i]))
Answer_data.close()

#正解数を表示.
flag = 0
for i in range (0,len(lines)):
    if output[i] == Ans[i]:
        flag = flag + 1
print("正解数", flag, "/", len(lines))
print("正解率", 100 * flag / len(lines), "%" )


# In[ ]:




