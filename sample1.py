#coding=utf-8

import cv2
recognizer = cv2.face.LBPHFaceRecognizer_create()         # �ҥΰV�m�H�y�ҫ���k
recognizer.read('face.yml')                               # Ū���H�y�ҫ���
cascade_path = 'C:/Users/User/.vscode/haarcascade_frontalface_default.xml'  # ���J�H�y�l�ܼҫ�
face_cascade = cv2.CascadeClassifier(cascade_path)        # �ҥΤH�y�l��

cap = cv2.VideoCapture(0)                                 # �}����v��
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
    faces = face_cascade.detectMultiScale(gray)  # �l�ܤH�y ( �ت��b��аO�X�~�� )

    # �إߩm�W�M id ����Ӫ�
    name = {
        '1':'Yichen',
        '2':'LiangYing',
        '3':'Orange',
        '4':'bad bitch',
      
        
    }

    # �̧ǧP�_�C�i�y�ݩ���� id
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)            # �аO�H�y�~��
        idnum,confidence = recognizer.predict(gray[y:y+h,x:x+w])  # ���X id ���X�H�ΫH�߫��� confidence
        if confidence < 100:
            text = name[str(idnum)]                               # �p�G�H�߫��Ƥp�� 60�A���o�������W�r
        else:
            text = '???'                                          # ���M�W�r�N�O ???
        cv2.putText(img, text, (x,y-5),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA) # �b�H�y�~�خǥ[�W�W�r

    cv2.imshow('oxxostudio', img)
    if cv2.waitKey(5) == ord('q'):
        break    # ���U q �䰱��
cap.release()
cv2.destroyAllWindows()