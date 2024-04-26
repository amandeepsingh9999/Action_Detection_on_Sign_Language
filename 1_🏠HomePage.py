import streamlit as st
import requests
from streamlit_lottie import st_lottie


def loadLottie(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


lottieCoding = loadLottie("https://lottie.host/ffa4239f-e7f0-4a7d-8bbb-ffef3555ee46/VYXBXQOqxo.json")
# st.title("Hello there :wave:")
# st.sidebar.image("Hwllio")
st.set_page_config(
    page_title="Action Detection App",
    page_icon=":wave:",
    layout="wide",
)
st.sidebar.title("👾 DJANGO")
# import train
with st.container():
    # st.header("Hii i'm Aastha Verma and Amandeep Singh")
    # st.subheader("We are here present our project - 👇")
    st.title("ACTION DETECTION ON SIGN LANGUAGE :wave:")
    st.caption('"_Although the world is full of suffering . It is also full of overcoming of it_"')
    st.caption("--Helen keller")
with st.container():
    st.write("---")
    col1, col2 = st.columns(2)
    with col1:
        st.write("##")
        st.write("##")
        st.header("INDIAN STANDARD SIGN LANGUAGE -->")
        st.write("##")
        st.write(
            '1- ISL uses both hands similar to British Sign Language and is similar to International Sign Language.')
        st.write('2- ISL alphabets derived from British Sign Language and French Sign Language alphabets')
        st.write('3- Unlike its american counterpart which uses one hand, uses both hands to represent alphabets.')
    with col2:
        st_lottie(lottieCoding, height=500)
st.sidebar.success("Select a page above")

# Step One --
st.write("---")
with st.container():
    col, col1 = st.columns(2)
    with col:
        st.header("**METHODOLOGY**")
        st.write("##")
        st.subheader("Step - 1")
        st.write("**First we will clearly define what we are trying to do** ")
        st.write("--")
        st.write("- So we are trying to create a web app that will help our friend that are specially disabled , "
                 "who cannot hear so they uses sign language so we will detect their hand gesture and make our "
                 "communication easier")
        st.write("- So we will start with American Standard Sign Language and create our data ")
    with col1:
        st_lottie(loadLottie("https://lottie.host/17fb39ce-5933-4b83-a83f-103a0594f3b1/z7Zt2WJ8eq.json"), height=400)

# Step Two -
with st.container():
    col, col1 = st.columns(2)
    with col:
        st_lottie(loadLottie("https://lottie.host/9e77aaf0-6a81-4059-9544-47534d3cd542/MXCM9SLf6s.json"), height=400)
    with col1:
        st.subheader("Step - 2")
        st.write("**Getting Right data** ")
        st.write("--")
        st.write("- So we will be creating our hand sign data based on ASL(American Standard Sign Language), "
                 "That will be our first step for Achieving ISL because ASL uses one hand gesture that will be easy "
                 "for us ")
        st.write("- After we have created our data based on ASL and training our model has been complete then only we "
                 "will be moving towards ISL")

# Step Three -
with st.container():
    col, col1 = st.columns(2)
    with col:
        st.subheader("Step - 3")
        st.write("**Figure out what matters --** ")
        st.write("--")
        st.write("- We'll look at the sign language data and figure out what parts are important for recognizing "
                 "gestures. This might include things like where the hands are moving and how fast they're moving.")
    with col1:
        st_lottie(loadLottie("https://lottie.host/0c9980d1-1d34-4093-9aa8-4aca23d67b2b/5cFfj8Z9Ty.json"), height=400)

# Step Four -
with st.container():
    col, col1 = st.columns(2)
    with col:
        st_lottie(loadLottie("https://lottie.host/b3d0d705-ca01-404d-a2ac-1274be3bccff/UgyNn14b8t.json"), height=400)
    with col1:
        st.subheader("Step - 4")
        st.write("**Training the machine** ")
        st.write("--")
        st.write("- We'll pick the best ways to teach the computer to recognize sign language gestures. This might "
                 "involve using fancy math and algorithms or just lots of examples.")
        st.write("- We have to use enough examples for our this process")

# Step Five -
with st.container():
    col, col1 = st.columns(2)
    with col:
        st.subheader("Step - 5")
        st.write("**Testing it out --** ")
        st.write("--")
        st.write("- Now we will test our trained model which is prepared on images we have clicked so now we will "
                 "test it out in out **test.py** ")
        st.write("- After we have tested our data on user input in ASL we will be continue to ISL data creation and "
                 "testing")
    with col1:
        st_lottie(loadLottie("https://lottie.host/584f3dd1-64da-444b-854e-4452de62d58e/nISb2BocKY.json"), height=400)

# what are the Steps we followed to achieve data collection for our Hand detector

st.write("---")
st.header("_:blue[CODE --]_")
st.subheader("IMPORTING THE LIBRARIES:- ")
code = '''
import math
import streamlit as st
import time
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np

'''
code1 = '''
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)

folder = "Images/I"
offset = 20
imageSize = 300
counter = 1

FramePlaceholder = st.empty()'''
code2 = '''
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

        FramePlaceholder.image(imageWhite, channels="BGR", caption='Processed Image', use_column_width=True)

    key = cv2.waitKey(1)

    if key == ord("s"):
        counter += 1
        cv2.imwrite(f"{folder}/images_{time.time()}.jpg", imageWhite)
'''
st.code(code, language='python')
st.subheader("PROVIDING THE VALUES :-")
st.code(code1, language='python')
st.subheader("WRITING THE MAIN CODE FOR COLLECTING THE IMAGE")
st.code(code2, language='python')