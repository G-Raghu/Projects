import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mpdraw = mp.solutions.drawing_utils
while True:
    su, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(imgRGB)
    
    if result.multi_hand_landmarks:
        for handlns in result.multi_hand_landmarks:
            mpdraw.draw_landmarks(img, handlns, mp_hands.HAND_CONNECTIONS)


    cv2.imshow("Image", img)
    cv2.waitKey(1)