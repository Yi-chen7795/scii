#coding=utf-8
"""
1.入館、出館時間
2.看要不要偵測到臉後直接關掉
3.找找看有沒有畫面不會馬上卡住的方法
"""
import datetime
import openpyxl
import cv2
import pyfirmata
import time

wb = openpyxl.load_workbook('scii.xlsx', data_only=True)  #開啟欲輸入資料的表格
s1 = wb['worksheet1']            

date = datetime.date.today()    #紀錄偵測時日期

recognizer = cv2.face.LBPHFaceRecognizer_create()         # 啟用訓練人臉模型
recognizer.read('face.yml')                               # 讀取我們剛剛訓練的模型資料
cascade_path = 'C:/Users/User/.vscode/haarcascade_frontalface_default.xml'  # 載入人臉追蹤模型(opencv的github提供)
face_cascade = cv2.CascadeClassifier(cascade_path)        # 啟用人臉追蹤

pin1 = 11                           #LED燈位子設定
pin2 = 12
pin3 = 13
port = 'COM3'
board = pyfirmata.Arduino(port)     #連接到Arduino

cap = cv2.VideoCapture(0)           # 開啟攝影機
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    ret, img = cap.read()
    if not ret:
        print("Cannot receive frame")
        break
    img = cv2.resize(img,(540,300))              # 縮小尺寸，加快辨識效率
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  # 轉換成黑白
    faces = face_cascade.detectMultiScale(gray)  # 追蹤人臉 ( 為了標記出外框 )

    # 建立欲顯示之id對應名稱
    name = {
        '1':'Yichen',
        '2':'LiangYing',
        '3':'Orange'
    }
    
    Yichen = 0              #每個人(判斷現在開關狀態)的初始值設為0(關閉)
    LiangYing = 0
    Orange = 0
    
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)            # 標記人臉外框
        idnum,mistake = recognizer.predict(gray[y:y+h,x:x+w])  # 取出 id 號碼以及辨識誤差值
        if mistake < 120:
            text = name[str(idnum)]                               # 如果誤差值小於120，取得對應的名字
            if idnum == 1 and Yichen == 0:                        # 當偵測到id=1且為關閉狀態時執行開燈
                board.digital[pin1].write(1) 
                time.sleep(5)
                s1['B3'].value = date                             # 於該姓名列填入入館日期
                Yichen = 1
                print("1 on")
            if idnum == 1 and Yichen == 1:                        # 當偵測到id=2且為開啟狀態時執行關燈
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
            text = '???'                                          # 偵測不出是誰時顯示 ???
        cv2.putText(img, text, (x,y-5),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA) # 在人臉外框旁加上名字
    
    wb.save('sciii.xlsx')               #儲存表格
    cv2.imshow('camera', img)
    if cv2.waitKey(5) == ord('q'):
        break    # 按下 q 鍵停止
cap.release()
cv2.destroyAllWindows()