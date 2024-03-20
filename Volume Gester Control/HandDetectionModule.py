import cv2
import mediapipe as mp
import time

class Handdector():
    def __init__(self,mode=False,max=2,detectioncon=0.5,trackcon=0.5):
        self.mode=mode
        self.max=max
        self.dc=detectioncon
        self.tc=trackcon

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()
        self.mpdraw = mp.solutions.drawing_utils
    
    def findhands(self,img,draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.result = self.hands.process(imgRGB)

        if self.result.multi_hand_landmarks:
            for handlns in self.result.multi_hand_landmarks:
                if draw:
                    self.mpdraw.draw_landmarks(img, handlns, self.mp_hands.HAND_CONNECTIONS)
        return img
    
    def findposition(self,img,handno=0,draw=True):
        lmlist = []
        if self.result.multi_hand_landmarks:
            myhand = self.result.multi_hand_landmarks[handno]
            for id,lm in enumerate(myhand.landmark):
                h,w,c=img.shape
                cx,cy=int(lm.x*w),int(lm.y*h)
                #print(id,cx,cy)
                lmlist.append([id,cx,cy])
                if draw:
                    cv2.circle(img,(cx,cy),10,(0,100,255),cv2.FILLED)
        return lmlist

def main():
    pT=0
    cap=cv2.VideoCapture(0)
    detector=Handdector()
    while True:
        success, img = cap.read()
        img=cv2.flip(img,1)

        detector.findhands(img)
        ls=detector.findposition(img)
        if len(ls) != 0:
            print(ls[4])

        cT=time.time()
        fps=1/(cT-pT)
        pT=cT
        
        cv2.putText(img,f"FPS:{int(fps)}",(10,50),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)
        cv2.imshow("Image",img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()
