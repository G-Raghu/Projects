import cv2
import pickle as pk
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
import csv
import time
from datetime import datetime

cap=cv2.VideoCapture(0)
detect=cv2.CascadeClassifier("data/haarcascade_frontalface_default.xml")

with open("data/faces.pkl","rb") as fle:
    FACES=pk.load(fle)

with open("data/names.pkl","rb") as fle:
    LABELS=pk.load(fle)

kn=KNeighborsClassifier(n_neighbors=5)
kn.fit(FACES,LABELS)

Columns=["Name","Time"]

while True:
    pop,img=cap.read()

    faces=detect.detectMultiScale(img,1.3,5)
    for (x,y,w,h) in faces:
        crop=img[y:y+h,x:x+w,:]
        resized=cv2.resize(crop,(50,50)).flatten().reshape(1,-1)
        output=kn.predict(resized)
        ct=time.time()
        date=datetime.fromtimestamp(ct).strftime("%d-%m-%Y")
        tim=datetime.fromtimestamp(ct).strftime("%H:%M:%S")
        attendance=[str(output[0]),str(tim)]
        pth=os.path.isfile("Attendance/Attendance_"+date+".csv")

        cv2.putText(img,str(output[0]),(x,y-10),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(255,255,255),1)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,250,0),3)
    cv2.imshow("Output",img)
    k=cv2.waitKey(1) 
    if k==ord('o'):
        if pth:
            with open("Attendance/Attendance_"+date+".csv","+a") as Csv:
                wt=csv.writer(Csv)
                wt.writerow(attendance)
            Csv.close()
        else:
            with open("Attendance/Attendance_"+date+".csv","+a") as Csv:
                wt=csv.writer(Csv)
                wt.writerow(Columns)
                wt.writerow(attendance)
            Csv.close()
    if k==ord('e'):
        break