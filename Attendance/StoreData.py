import cv2
import pickle as pk
import numpy as np
import os

cap=cv2.VideoCapture(0)
detect=cv2.CascadeClassifier("data/haarcascade_frontalface_default.xml")

faces_data=[]

name=input("Enter your name")

while True:
    pop,img=cap.read()
     
    faces=detect.detectMultiScale(img,1.3,5)
    for (x,y,w,h) in faces:
        crop=img[y:y+h,x:x+w,:]
        resized=cv2.resize(crop,(50,50))
        if len(faces_data)<=100:
            faces_data.append(resized)
        cv2.putText(img,str(len(faces_data)),(50,100),cv2.FONT_HERSHEY_COMPLEX,2,(250,0,0),2)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,250,0),3)
    cv2.imshow("Output",img)
    k=cv2.waitKey(1) 
    if k==ord('e') or len(faces_data)==100:
        break

faces_data=np.asarray(faces_data)
faces_data=faces_data.reshape(100,-1)

if "names.pkl" not in os.listdir('data/'):
    names=[name]*100
    with open("data/names.pkl","wb") as fle:
        pk.dump(names,fle)
else:
    with open("data/names.pkl","rb") as fle:
       names=pk.load(fle)
    names=names+[name]*100
    with open("data/names.pkl","wb") as fle:
        pk.dump(names,fle)


if "faces.pkl" not in os.listdir('data/'):
    with open("data/faces.pkl","wb") as fle:
        pk.dump(faces_data,fle)
else:
    with open("data/faces.pkl","rb") as fle:
       faces=pk.load(fle)
    faces=np.append(faces,faces_data,axis=0)
    with open("data/faces.pkl","wb") as fle:
        pk.dump(faces_data,fle)