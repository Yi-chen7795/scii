#coding=utf-8
"""
1.�J�]�B�X�]�ɶ�
2.�ݭn���n�������y�᪽������
3.���ݦ��S���e�����|���W�d����k
"""
import datetime
import openpyxl
import cv2
import pyfirmata
import time

wb = openpyxl.load_workbook('scii.xlsx', data_only=True)  #�}�ұ���J��ƪ����
s1 = wb['worksheet1']            

date = datetime.date.today()    #���������ɤ��

recognizer = cv2.face.LBPHFaceRecognizer_create()         # �ҥΰV�m�H�y�ҫ�
recognizer.read('face.yml')                               # Ū���ڭ̭��V�m���ҫ����
cascade_path = 'C:/Users/User/.vscode/haarcascade_frontalface_default.xml'  # ���J�H�y�l�ܼҫ�(opencv��github����)
face_cascade = cv2.CascadeClassifier(cascade_path)        # �ҥΤH�y�l��

pin1 = 11                           #LED�O��l�]�w
pin2 = 12
pin3 = 13
port = 'COM3'
board = pyfirmata.Arduino(port)     #�s����Arduino

cap = cv2.VideoCapture(0)           # �}����v��
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    ret, img = cap.read()
    if not ret:
        print("Cannot receive frame")
        break
    img = cv2.resize(img,(540,300))              # �Y�p�ؤo�A�[�ֿ��ѮĲv
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  # �ഫ���¥�
    faces = face_cascade.detectMultiScale(gray)  # �l�ܤH�y ( ���F�аO�X�~�� )

    # �إ߱���ܤ�id�����W��
    name = {
        '1':'Yichen',
        '2':'LiangYing',
        '3':'Orange'
    }
    
    Yichen = 0              #�C�ӤH(�P�_�{�b�}�����A)����l�ȳ]��0(����)
    LiangYing = 0
    Orange = 0
    
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)            # �аO�H�y�~��
        idnum,mistake = recognizer.predict(gray[y:y+h,x:x+w])  # ���X id ���X�H�ο��ѻ~�t��
        if mistake < 120:
            text = name[str(idnum)]                               # �p�G�~�t�Ȥp��120�A���o�������W�r
            if idnum == 1 and Yichen == 0:                        # ������id=1�B���������A�ɰ���}�O
                board.digital[pin1].write(1) 
                time.sleep(5)
                s1['B3'].value = date                             # ��өm�W�C��J�J�]���
                Yichen = 1
                print("1 on")
            if idnum == 1 and Yichen == 1:                        # ������id=2�B���}�Ҫ��A�ɰ������O
                board.digital[pin1].write(0)                      
                time.sleep(5)
                Yichen = 0
                print("1 off")
            if idnum == 2 and LiangYing == 0:
                board.digital[pin2].write(1)
                time.sleep(5)
                s1['B2'].value = date
                LiangYing = 1
                print("2 on")
            if idnum == 1 and LiangYing == 1:
                board.digital[pin2].write(0)
                time.sleep(5)
                LiangYing = 0
                print("2 off")
            if idnum == 3 and Orange == 0:
                board.digital[pin3].write(1)
                time.sleep(5)
                s1['B4'].value = date
                Orange = 1
                print("3 on")
            if idnum == 1 and Orange == 1:
                board.digital[pin3].write(0)
                time.sleep(5)
                Orange = 0
                print("3 off")
                
        else:
            text = '???'                                          # �������X�O�֮���� ???
        cv2.putText(img, text, (x,y-5),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA) # �b�H�y�~�خǥ[�W�W�r
    
    wb.save('sciii.xlsx')               #�x�s���
    cv2.imshow('camera', img)
    if cv2.waitKey(5) == ord('q'):
        break    # ���U q �䰱��
cap.release()
cv2.destroyAllWindows()