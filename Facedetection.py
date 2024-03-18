import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpFace = mp.solutions.face_detection
mpdraw = mp.solutions.drawing_utils
fd = mpFace.FaceDetection(0.80)


pt=0
ct=0
while True:
    success, img = cap.read()
    
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = fd.process(imgRGB)

    if results.detections:
        for id,detec in enumerate(results.detections):
            print(id,detec)    
            #mpdraw.draw_detection(img, detec)
            bboxC = detec.location_data.relative_bounding_box
            h,w,c=img.shape
            bbox = int(bboxC.xmin * w),int(bboxC.ymin * h), \
                   int(bboxC.width * w),int(bboxC.height * h)
            cv2.rectangle(img, bbox, (225,0,255),2)

    ct=time.time()
    fps=1/(ct-pt)
    pt=ct
    cv2.putText(img,f"{int(detec.score[0]*100)}%",(bbox[0],bbox[1]-20),cv2.FONT_HERSHEY_PLAIN,2,(25,100,100),2)
    cv2.imshow("Image", img)
    cv2.waitKey(1)