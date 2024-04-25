import streamlit as st
import cv2
import numpy as np
from PIL import Image
from MyModel import MyModel

model = MyModel()

label2text = {
    'aleff': 'aleff - أ',
    'zay': 'zay - ز',
    'seen': 'seen - س',
    'sheen': 'sheen - ش',
    'saad': 'saad - ص',
    'dhad': 'dhad - ض',
    'taa': 'tah - ط',
    'dha': 'dhaa - ظ',
    'ain': 'ain - ع',
    'ghain': 'ghain - غ',
    'fa': 'faa - ف',
    'bb': 'baa - ب',
    'gaaf': 'qaaf - ق',
    'kaaf': 'kaaf - ك',
    'laam': 'laam - ل',
    'meem': 'meem - م',
    'nun': 'noon - ن',
    'ha': 'haa - ه',
    'waw': 'waw - و',
    'yaa': 'ya - ئ',
    'toot': 'taa marbouta - ة',
    'al': 'al - لا',
    'ta': 'taa - ت',
    'la': 'la - ال',
    'ya': 'yaa - ى',
    'thaa': 'thaa - ث',
    'jeem': 'jeem - ج',
    'haa': 'haa - ح',
    'khaa': 'khaa - خ',
    'dal': 'dal - د',
    'thal': 'thal - ذ',
    'ra': 'raa - ر'
}

label2image = {}
for label in label2text.keys():
    imgname = label + '.png'
    imgpath = '../assets/image_lables/' + imgname
    label2image[label] = Image.open(imgpath)

cap = cv2.VideoCapture(0)

st.title("Arabic Sign Language Recognition")

try:
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        label, prop = model.predict([frame])
        if label[0] != '-1' and prop[0] > 0.5:
            image = label2image[label[0]]
            frame = putImage(frame, image)
        else:
            cv2.putText(frame, "Please Sign a Letter", (150, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 2, cv2.LINE_AA)
        
        st.image(frame, channels="BGR", caption='Letters of Arabic Sign Language Recognition')

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
except Exception as e:
    cap.release()
    raise e
