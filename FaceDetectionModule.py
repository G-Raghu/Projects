import cv2
import mediapipe as mp
import time

class FaceDetector:
    def __init__(self):
        self.mpFace = mp.solutions.face_detection
        self.mpdraw = mp.solutions.drawing_utils
        self.fd = self.mpFace.FaceDetection(0.80)
    
    def FindFace(self,img):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.fd.process(imgRGB)
        bboxs=[]

        if self.results.detections:
            for id,detec in enumerate(self.results.detections):
                print(id,detec)    
                #mpdraw.draw_detection(img, detec)
                bboxC = detec.location_data.relative_bounding_box
                h,w,c=img.shape
                bbox = int(bboxC.xmin * w),int(bboxC.ymin * h), \
                       int(bboxC.width * w),int(bboxC.height * h)
                bboxs.append([id,bbox,detec.score])

                cv2.rectangle(img, bbox, (225,0,255),2)
                cv2.putText(img,f"{int(detec.score[0]*100)}%",(bbox[0],bbox[1]-20),cv2.FONT_HERSHEY_PLAIN,2,(25,100,100),2)
        return img,bboxs

def main():
    cap = cv2.VideoCapture(0)
    pt=0
    detector=FaceDetector()
    while True:
        success, img = cap.read()
        img,bboxs=detector.FindFace(img)

        ct=time.time()
        fps=1/(ct-pt)
        pt=ct
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
