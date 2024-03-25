import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import playsound
import mediapipe as mp
from tensorflow import keras
import av

mp_holistic = mp.solutions.holistic 
mp_drawing = mp.solutions.drawing_utils

modelF= keras.models.load_model('new_action_mod.h5')
st.set_page_config(page_title="Streamlit WebRTC Demo", page_icon="🖐")





class OpenCamera (VideoProcessorBase):
    def __init__(self) -> None :
        self.sequence = []
        self.sentence = []
        self.threshold = 0.8
        self.actions = np.array(['Ahmar', 'Abdyad', 'Akhdar'])

    


    def mediapipe_detection(self,image, model):
        self.image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.image.flags.writeable = False                 
        self.results = model.process(image)                
        self.image.flags.writeable = True                   
        self.image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image, self.results


    # Function to draw landmarks with styling
    def draw_styled_landmarks(self,image, results):

    
            mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, 
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)) 
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=2), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)) 
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=2), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)) 
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)) 


    def extract_keypoints(self, results):
        self.key1 = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
        self.key2 = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
        self.lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        self.rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        return np.concatenate([self.key1, self.key2, self.lh, self.rh])
    

    # Visualize prediction:

    # Define functions for visualization and detection

    
    def recv(self, frame):
        self.predictions=[]
        img=frame.to_ndarray(format="bgr24")
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            image, results = self.mediapipe_detection(img,holistic)
            self.draw_styled_landmarks(image, results)
            # 2. Prediction logic
            keypoints = self.extract_keypoints(results)

            self.sequence.append(keypoints)
            self.sequence = self.sequence[-30:]
              
            if len(self.sequence) == 30:
                    res = modelF.predict(np.expand_dims(self.sequence, axis=0))[0]
                    self.predictions.append(np.argmax(res))

                    if len(self.predictions) >= 15:
                        unique_predictions = np.unique(self.predictions[-15:])
                        if unique_predictions.size > 0:
                            most_common_prediction = unique_predictions[0]
                            if most_common_prediction == np.argmax(res) and res[np.argmax(res)] > self.threshold:
                                current_action = self.actions[np.argmax(res)]
                                if len(self.sentence) == 0 or current_action != self.sentence[-1]:
                                    self.sentence.append(current_action)
                                    audio_files = {'Abyad': 'audio/abyad.mp3', 'Ahmar': 'audio/ahmar.mp3', 'Akhdar': 'audio/akhdar.mp3'}
                                    if current_action in audio_files:
                                        playsound(audio_files[current_action])
                                     

                                        if len(self.sentence) > 1:
                                             self.sentence = self.sentence[-1:]     
                   #image = prob_viz(res, self.actions, image, colors)
            #3. Viz logic
            colors = [(245,117,16), (117,245,16), (16,117,245)]
                    # Viz probabilities
    
            
            #cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(self.sentence), (3,30),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
    #   av.VideoFrame.from_ndarray(image, format="bgr24")      
        return av.VideoFrame.from_ndarray(image, format="bgr24")

st.title('ICAN - Lebanese Sign Language Interpreter')
st.write('Made by ICAN Team')

st.header('Real-Time Hand Gesture Recognition Using Mediapipe & LSTM')
st.markdown('To start detecting your LSL gesture click on the "START" button')
ctx = webrtc_streamer(
    key="example",
    video_processor_factory=OpenCamera,
    rtc_configuration={ # Add this line
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }, media_stream_constraints={"video": True, "audio": False,}, async_processing=True
)
