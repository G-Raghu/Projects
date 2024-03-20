import cv2
import mediapipe as mp
import time
import HandDetectionModule as hm #importing HandDetectionModule file
import math
import numpy as np

# installing and importiong pycaw library
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

pT=0
cap=cv2.VideoCapture(0)

#calling Handdetector class from HandDetectionModule
detector=hm.Handdetector() 

# connnecting to speakers
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
#volume.GetMute()
#volume.GetMasterVolumeLevel()
volrange=volume.GetVolumeRange() # storing volumerange

minvol=volrange[0]
maxvol=volrange[1]

while True:
    success, img = cap.read()
    img=cv2.flip(img,1)

    detector.findhands(img) # calling findhands funtion to detect hands
    ls=detector.findposition(img) # calling findposition and storing the id,position of hand in ls list
    if len(ls) != 0:
        x1,y1=ls[4][1],ls[4][2] #storing id and possition of thumb finger in x1 and y1
        x2,y2=ls[8][1],ls[8][2] #storing id and possition of index finger in x1 and y1
        cx,cy=(x1+x2)//2,(y1+y2)//2 #storing midpoint x1,x2 and y1,y2 in cx and cy

        cv2.circle(img,(x1,y1),15,(250,0,250),cv2.FILLED) #drawing circle to thumb finger
        cv2.circle(img,(x2,y2),15,(250,0,250),cv2.FILLED) #drawing circle to index finger
        cv2.line(img,(x1,y1),(x2,y2),(250,0,250),3) #connecting thumb and index finger
        cv2.circle(img,(cx,cy),15,(250,0,250),cv2.FILLED) #drawing circle to the mid point of line connecting infex and thumb finger    

        length=math.hypot((x2-x1),(y2-y1)) #finding the length of the line
        #print(length)

        vol=np.interp(length,[50,250],[minvol,maxvol]) #interlinking the  minimum and maximum values of length and volume 
        print(int(length),vol)
        volume.SetMasterVolumeLevel(vol, None) #contorling volume on basis of vol

        if length<=50:
            cv2.circle(img,(cx,cy),15,(0,250,0),cv2.FILLED)


    cT=time.time()
    fps=1/(cT-pT)
    pT=cT
        
    cv2.putText(img,f"FPS:{int(fps)}",(10,50),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)
    cv2.imshow("Image",img)
    cv2.waitKey(1)



