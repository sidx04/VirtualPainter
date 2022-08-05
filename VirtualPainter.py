import cv2
import numpy as np
import time
import os
import HandTrackingModule as htm

folderpath = "Header"
myList = os.listdir(folderpath)
print(myList)
overlayList = []

for imPath in myList:
    image = cv2.imread(f'{folderpath}/{imPath}')
    overlayList.append(image)
print(len(overlayList))

header = overlayList[0]
drawColor = (255, 0, 255)

cap = cv2.VideoCapture(1)
cap.set(3, 1280)
cap.set(4, 720)

detector = htm.handDetector(detectionCon=0.7, maxHands=1)
xp, yp = 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

while True:

    # import image
    success, img = cap.read()
    img = cv2.flip(img, 1)

    # find hand landmarks
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]
        fingers = detector.fingersUp()

    img[0:65, 0:640]=header
    cv2.imshow("Image", img)
    cv2.imshow("Canvas", imgCanvas)
    cv2.waitKey(1)
