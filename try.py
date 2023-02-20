#coding=utf-8

import cv2
import numpy as np
path ='C:/Users/User/.vscode/haarcascade_frontalface_default.xml'  # ���J�H�y�l�ܼҫ�(opencv��github����)
detector = cv2.CascadeClassifier(path)            # ���J�H�y�l�ܼҫ�
recog = cv2.face.LBPHFaceRecognizer_create()      # �ҥΰV�m�H�y�ҫ�
faces = []   # �x�s�H�y��m�j�p����C
ids = []     # �O���ӤH�y id ����C

print('Start!')
print('wait for a minute...')

for i in range(505):
    img = cv2.imread("C:/Users/User/sci/20805/%d.jpg"%(i))           # �̧Ƕ}�Ҹ�Ƨ���class 1���Ӥ�
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # ��m�ഫ���¥�
    img_np = np.array(gray,'uint8')               # �ഫ�����w�s�X�� numpy �}�C
    face = detector.detectMultiScale(gray)        # �^���H�y�ϰ�
    for(x,y,w,h) in face:
        faces.append(img_np[y:y+h,x:x+w])         # �O���H�y����m�M�j�p���������ƭ�
        ids.append(1)                             # �O���H�y������id�A�u��O��ơA���O1���class 1��id��1

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


              

print('training...')                              # ���ܶ}�l�V�m
recog.train(faces,np.array(ids))                  # �}�l�V�m
recog.save('face.yml')                            # �V�m�����x�s�� face.yml
print('ok!')