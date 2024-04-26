import math
import streamlit as st
import time
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import os

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)

folder = "Images/iloveyou"
offset = 20
imageSize = 300
counter = 1

FramePlaceholder = st.empty()

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imageWhite = np.ones((imageSize, imageSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        imgCropShape = imgCrop.shape
        # imageWhite[0:imgCropShape[0], 0:imgCropShape[1]] = imgCrop

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imageSize / h
            wCal = math.ceil(k * w)
            imageResize = cv2.resize(imgCrop, (wCal, imageSize))
            imageResizeShape = imageResize.shape
            wGap = math.ceil((imageSize - wCal) / 2)
            imageWhite[:, wGap:wCal + wGap] = imageResize
        else:
            k = imageSize / w
            hCal = math.ceil(k * h)
            imageResize = cv2.resize(imgCrop, ( imageSize, hCal))
            imageResizeShape = imageResize.shape
            hGap = math.ceil((imageSize - hCal) / 2)
            imageWhite[hGap:hCal + hGap, :] = imageResize

        cv2.imshow("ImageCrop", imgCrop)
        # imageWhite = cv2.cvtColor(imageWhite, cv2.COLOR_BGR2RGB)
        cv2.imshow("WhiteImage", imageWhite)
        FramePlaceholder.image(img, channels="BGR", caption='Processed Image', use_column_width=True)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)

    if key == ord("s"):
        counter += 1
        cv2.imwrite(f"{folder}/images_{time.time()}.jpg", imageWhite)
        print(counter)

