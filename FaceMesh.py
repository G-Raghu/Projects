import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)


mpdraw = mp.solutions.drawing_utils
mp_facemesh = mp.solutions.face_mesh
face = mp_facemesh.FaceMesh(max_num_faces=2)
while True:
    su, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = face.process(imgRGB)
    
    if result.multi_face_landmarks:
        for facelns in result.multi_face_landmarks:
            mpdraw.draw_landmarks(img, facelns, mp_facemesh.FACE_CONNECTIONS)


    cv2.imshow("Image", img)
    cv2.waitKey(1)