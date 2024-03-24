import streamlit as st
import cv2
import numpy as np
import os
import time
import mediapipe as mp
import pyttsx3
from playsound import playsound
import tensorflow as tf
import warnings
from keras.models import load_model
warnings.filterwarnings('ignore')



# Load model:
model = load_model('new_action_mod.h5')

model.load_weights('new_action_mod.h5')
# MP Holistic:
mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

DATA_PATH = os.path.join('MP_Data_1') 

actions = np.array(['Ahmar', 'Abyad', 'Akhdar'])

# Thirty videos worth of data
no_sequences = 30

# Videos are going to be 30 frames in length
sequence_length = 30

# Folder start
start_folder = 1
for action in actions:
    action_path = os.path.join(DATA_PATH, action)

    # Check if the directory exists
    if os.path.exists(action_path):
        # Get the list of files in the directory
        files = os.listdir(action_path)

        # Check if the list is not empty
        if files:
            # Convert the file names to integers and find the maximum
            dirmax = np.max(np.array(files).astype(int))
        else:
            dirmax = 0  # Set a default value if the directory is empty
    else:
        dirmax = 0  # Set a default value if the directory does not exist

    for sequence in range(1, no_sequences + 1):
        try:
            os.makedirs(os.path.join(DATA_PATH, action, str(dirmax + sequence)))
        except:
            pass



def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def draw_styled_landmarks(image, results):
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             ) 
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             ) 

# Extract Keypoint values
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])



# Visualize prediction:
def prob_viz(res, actions, input_frame):
    output_frame = input_frame.copy()

    pred_dict = dict(zip(actions, res))
    # sorting for prediction and get top 5
    prediction = sorted(pred_dict.items(), key=lambda x: x[1])[::-1][:5]

    for num, pred in enumerate(prediction):
        text = '{}: {}'.format(pred[0], round(float(pred[1]),4))
        # cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, text, (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2, cv2.LINE_AA) 
    return output_frame


###############################################################################################
                                            # STREAMLIT #

col1, col2 = st.columns((3,1))
with col1:
    st.title('ICAN - Lebanese Sign Language Interpreter ')
    st.write('Make by ICAN Team')

with col2:
    st.image('Logo.png')

# Checkboxes
st.header('Webcam')

col1, col2, col3 = st.columns(3)
with col1:
    show_webcam = st.checkbox('Show webcam')

# Webcam
FRAME_WINDOW = st.image([])
cap = cv2.VideoCapture(-1) # device 1/2
def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized

def start_detection(st, stframe):
    cap = cv2.VideoCapture(-1)
    sequence = []
    sentence = []
    predictions = []
    threshold = 0.4

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            image, results = mediapipe_detection(frame, holistic)

            # Check if either left or right hand landmarks are detected
            if results.left_hand_landmarks or results.right_hand_landmarks:
                draw_styled_landmarks(image, results)

                keypoints = extract_keypoints(results)
                sequence.append(keypoints)
                sequence = sequence[-30:]

                if len(sequence) == 30:
                    res = model.predict(np.expand_dims(sequence, axis=0))[0]
                    predictions.append(np.argmax(res))

                    if len(predictions) >= 10:
                        unique_predictions = np.unique(predictions[-10:])
                        if unique_predictions.size > 0:
                            most_common_prediction = unique_predictions[0]
                            if most_common_prediction == np.argmax(res) and res[np.argmax(res)] > threshold:
                                current_action = actions[np.argmax(res)]
                                if len(sentence) == 0 or current_action != sentence[-1]:
                                    sentence.append(current_action)
                                    audio_files = {'Abyad': 'audio/abyad.mp3', 'Ahmar': 'audio/ahmar.mp3', 'Akhdar': 'audio/akhdar.mp3'}
                                    if current_action in audio_files:
                                        playsound(audio_files[current_action])

                                        if len(sentence) > 5:
                                             sentence = sentence[-5:]

                    # image = prob_viz(res, actions, image, colors)

                cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
                cv2.putText(image, ' '.join(sentence), (3, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Display the frame in the Streamlit app
            frame = cv2.resize(image, (0, 0), fx=0.8, fy=0.8)
            frame = image_resize(image=frame, width=640)
            stframe.image(frame, channels='BGR', use_column_width=True)

            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


cap.release()
cv2.destroyAllWindows()



st.markdown(' ## Output')


stframe = st.empty()
if show_webcam:
    vid = cv2.VideoCapture(0)
    while True:
        ret, img = vid.read()
        if ret is None:
            print("Error: Unable to read frame from webcam")
            break
        img = cv2.flip(img, 1)
        h, w, c = img.shape
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img)

        img.flags.writeable = True
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                    # Call the start_detection function with st and stframe arguments
        start_detection(st, stframe)

                    # Break the loop if 'q' is pressed or any other termination condition
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
elif not show_webcam:
    st.write('Please check the webcam box to start the interpreter')
