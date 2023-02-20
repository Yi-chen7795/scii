#coding=utf-8

import cv2
import numpy as np
path ='C:/Users/User/.vscode/haarcascade_frontalface_default.xml'  # 載入人臉追蹤模型(opencv的github提供)
detector = cv2.CascadeClassifier(path)            # 載入人臉追蹤模型
recog = cv2.face.LBPHFaceRecognizer_create()      # 啟用訓練人臉模型
faces = []   # 儲存人臉位置大小的串列
ids = []     # 記錄該人臉 id 的串列

print('Start!')
print('wait for a minute...')

for i in range(505):
    img = cv2.imread("C:/Users/User/sci/20805/%d.jpg"%(i))           # 依序開啟資料夾內class 1的照片
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 色彩轉換成黑白
    img_np = np.array(gray,'uint8')               # 轉換成指定編碼的 numpy 陣列
    face = detector.detectMultiScale(gray)        # 擷取人臉區域
    for(x,y,w,h) in face:
        faces.append(img_np[y:y+h,x:x+w])         # 記錄人臉的位置和大小內像素的數值
        ids.append(1)                             # 記錄人臉對應的id，只能是整數，都是1表示class 1的id為1

for i in range(505):
    img = cv2.imread("C:/Users/User/sci/20803/%d.jpg"%(i))          #class 2
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
    img_np = np.array(gray,'uint8')               
    face = detector.detectMultiScale(gray)        
    for(x,y,w,h) in face:
        faces.append(img_np[y:y+h,x:x+w])         
        ids.append(2)                             
        
for i in range(514):
    img = cv2.imread("C:/Users/User/sci/20631/%d.jpg"%(i))           #class 3
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
    img_np = np.array(gray,'uint8')               
    face = detector.detectMultiScale(gray)        
    for(x,y,w,h) in face:
        faces.append(img_np[y:y+h,x:x+w])         
        ids.append(3)                             


              

print('training...')                              # 提示開始訓練
recog.train(faces,np.array(ids))                  # 開始訓練
recog.save('face.yml')                            # 訓練完成儲存為 face.yml
print('ok!')