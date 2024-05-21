import streamlit as st
import cv2
import numpy as np
from tensorflow import keras
import mediapipe as mp
import random
from playsound import playsound

# Load the gesture recognition model
#actions = np.array(['Bird', 'Bee', 'Rabbit', 'Snake', 'Fish', 'Cat'])

# Load the ASL model
#asl_model = keras.models.load_model('new_action_mod.h5')
#asl_model.load_weights('new_action_mod.h5')
asl_actions = np.array(['Ahmar', 'Abyad', 'Akhdar'])
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
# Set up Mediapipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
colors = [(245, 221, 173), (245, 185, 265), (146, 235, 193),
          (204, 152, 295), (255, 217, 179), (0, 0, 179)]

# Function to perform MediaPipe detection
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

# Function to draw landmarks with styling
def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                               mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                               mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                               mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                               mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2))
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                               mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                               mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2))
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                               mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                               mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

# Function to extract keypoints from MediaPipe results
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

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
# Start detection function
def start_gesture_detection(st, stframe, model, actions):
    #actions = np.array(['Hello', 'Good', 'I', 'Home', 'Love'])
    #actions = np.array(['white', 'orange', 'no_acion', 'green', 'blue'])
    cap = cv2.VideoCapture(0)
    sequence = []
    sentence = []
    predictions = []
    threshold = 0.55

    # Set up Mediapipe
    mp_holistic = mp.solutions.holistic

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image, results = mediapipe_detection(frame, holistic)

            # Check if either left or right hand landmarks are detected
            if results.left_hand_landmarks or results.right_hand_landmarks:
                draw_styled_landmarks(image, results)

                keypoints = extract_keypoints(results)
                sequence.append(keypoints)
                sequence = sequence[-30:]


                # Draw green square
                # square_top_left = (50, 150)  # Lowered position
                # square_bottom_right = (300, 400)  # Larger size
                # cv2.rectangle(image, square_top_left, square_bottom_right, (0, 255, 0), thickness=2)

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
                                    # Remove "Unknown" from sentence if present
                                    if "Unknown" in sentence:
                                        sentence.remove("Unknown")
                                    sentence.append(current_action)


                                    audio_files = {'Hello': 'audio/Hello.mp3', 'Good': 'audio/Good.mp3', 'I':'audio/Me.mp3', 'Me':'audio/Me.mp3', 'Home':'audio/Home.mp3', 'Love':'audio/Love.mp3',
                                    'Bird': 'audio/Bird.mp3', 'Bee': 'audio/Bee.mp3', 'Snake':'audio/Snake.mp3', 'Rabit':'audio/Rabit.mp3', 'Fish':'audio/Fish.mp3','Cat':'audio/Cat.mp3',\
                                    'white': 'audio/abyad.mp3', 'orange': 'audio/Orange.mp3', 'blue':'audio/Blue.mp3', 'green':'audio/akhdar.mp3',
                                    'Canada': 'audio/Canada.mp3', 'Europe': 'audio/Europe.mp3', 'France':'audio/France.mp3','Lebanon':'audio/Lebanon.mp3'}
                                    if current_action in audio_files:
                                        playsound(audio_files[current_action])

                                        if len(sentence) > 1:
                                            sentence = sentence[-1:]
            else:
                sequence = []
                sentence = []

                # If no hand landmarks are detected, reset predictions and display "Unknown"
                predictions = []
                sentence.append("Unknown")

                # Display prompt to perform a sign
                cv2.putText(image, 'Please do a sign', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Display the recognized signs
            if results.left_hand_landmarks or results.right_hand_landmarks:
                cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
                cv2.putText(image, ' '.join(sentence), (3, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Display the frame in the Streamlit app
            frame = cv2.resize(image, (0, 0), fx=0.8, fy=0.8)
            frame = image_resize(image=frame, width=640)
            stframe.image(frame, channels='BGR', use_column_width=True)
            # Break the loop
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break


        cap.release()
        cv2.destroyAllWindows()

# Function to start ASL detection
def start_asl_detection():
    cap = cv2.VideoCapture(0)
    sequence = []
    current_action = ""
    threshold = 0.95
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                annotated_frame = frame.copy()
                frame, results = mediapipe_detection(frame, holistic)
                draw_styled_landmarks(frame, results)
                keypoints = extract_keypoints(results)
                sequence.append(keypoints)
                sequence = sequence[-30:]

                if len(sequence) == 30:
                    res = asl_model.predict(np.expand_dims(sequence, axis=0))[0]
                    if np.max(res) >= threshold:
                        action_index = np.argmax(res)
                        action = asl_actions[action_index]
                        current_action = action
                    sequence = []

                cv2.putText(annotated_frame, f'Current Action: {current_action}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.imshow('Learn LSL', annotated_frame)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
            else:
                break
        cap.release()
        cv2.destroyAllWindows()

# Streamlit UI
def main():
    st.title("Gesture and LSL Detection")
    st.write("""
    ## Gesture and LSL Detection Using Streamlit and MediaPipe
    """)
    option = st.sidebar.selectbox(
        'Choose the Detection Type:',
        ('Gesture Detection', 'Learning LSL')
    )

    if option == 'Gesture Detection':
        st.write('## Gesture Detection')
        st.write('Use this option to detect gestures.')
        show_webcam = st.checkbox('Show webcam')
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
                #added
                actions = np.array(['Hello', 'Good', 'I', 'Home', 'Love'])
                model = keras.models.load_model('greetingsCNN.h5', compile=False)



                            # Call the start_detection function with st and stframe arguments
                start_gesture_detection(st, stframe,model,actions)

                            # Break the loop if 'q' is pressed or any other termination condition
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
        elif not show_webcam:
            cv2.destroyAllWindows()
            st.write('Please check the webcam box to start the interpreter')


    elif option == 'Learning LSL':
        st.write('## Lebanese Sign Language Learning Section')
        st.write('Use this option to select a category to learn gestures.')
        category = st.selectbox(
            'Select a category:',
            ('Greetings', 'Animals', 'Colors', 'Alphabet', 'Countries')
        )

        if category == 'Greetings':
            st.write('### Greetings')
            st.write('Here you can learn common greetings in Lebanese Sign Language.')
            gesture_videos = [("Hello", "Videos/Hello.mp4"), ("Good", "Videos/Good.mp4"), ("I", "Videos/I.mp4"),("Love", "Videos/Love.mp4")]
            selected_index = st.slider("Select a video", 0, len(gesture_videos) - 1, 0)

            video_name, video_path = gesture_videos[selected_index]

            st.write(f"**{video_name}**")
            st.video(video_path)
            stframe = st.empty()
            testing = False

            if not testing:
                if st.button("Test your knowledge"):
                    testing = True
                    st.button("Stop the test")
                    vid = cv2.VideoCapture(0)
                    while testing:
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
                        actions = np.array(['Hello', 'Good', 'I', 'Home', 'Love'])
                        model = keras.models.load_model('greetingsCNN.h5', compile=False)



                            # Call the start_detection function with st and stframe arguments
                        start_gesture_detection(st, stframe,model,actions)


                        # Break the loop if 'q' is pressed or any other termination condition
                        if cv2.waitKey(10) & 0xFF == ord('q'):
                            break
                    vid.release()
                    cv2.destroyAllWindows()
                    testing = False
            else:
                if st.button("Stop the test"):
                    testing = False
                    st.button("Test your knowledge")
                    # Close webcam
                    vid.release()
                    cv2.destroyAllWindows()



        elif category == 'Colors':
            st.write('### Colors')
            st.write('Here you can learn common Colors in Lebanese Sign Language.')
            gesture_videos = [("White", "videos/White.mp4"), ("Orange", "videos/Orange.mp4"), ("Blue", "videos/BLue.mp4"), ("Green", "videos/Green.mp4")]
            selected_index = st.slider("Select a video", 0, len(gesture_videos) - 1, 0)

            video_name, video_path = gesture_videos[selected_index]         
            video_name, video_path = gesture_videos[selected_index]

            st.write(f"**{video_name}**")
            st.video(video_path)
            
            stframe = st.empty()
            testing = False

            if not testing:
                if st.button("Test your knowledge"):
                    testing = True
                    st.button("Stop the test")
                    vid = cv2.VideoCapture(0)
                    while testing:
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
                        actions = np.array(['white', 'orange', 'no_action', 'green','blue'])
                        model = keras.models.load_model('joellecolorsCNN.h5', compile=False)



                            # Call the start_detection function with st and stframe arguments
                        start_gesture_detection(st, stframe,model,actions)


                        # Break the loop if 'q' is pressed or any other termination condition
                        if cv2.waitKey(10) & 0xFF == ord('q'):
                            break
                    vid.release()
                    cv2.destroyAllWindows()
                    testing = False
        elif category == 'Animals':
            st.write('### Animals')
            st.write('Here you can learn common animals in Lebanese Sign Language.')
            
            gesture_videos = [("bee", "videos/Bee.mp4"), ("bird", "videos/Bird.mp4"), ("cat", "videos/Bee.mp4"), ("Fish", "videos/Fish.mp4"), ("Snake", "videos/Snake.mp4"), ("Rabbit", "videos/Rabbit.mp4")]
            selected_index = st.slider("Select a video", 0, len(gesture_videos) - 1, 0)

            video_name, video_path = gesture_videos[selected_index]

            st.write(f"**{video_name}**")
            st.video(video_path)
            
            stframe = st.empty()
            testing = False

            if not testing:
                if st.button("Test your knowledge"):
                    testing = True
                    st.button("Stop the test")
                    vid = cv2.VideoCapture(0)
                    while testing:
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
                        actions = np.array(['Bee', 'Bird', 'Cat','Fish','Rabit', 'Snake' ])
                        model = keras.models.load_model('animalsCNN1.h5', compile=False)



                            # Call the start_detection function with st and stframe arguments
                        start_gesture_detection(st, stframe,model,actions)


                        # Break the loop if 'q' is pressed or any other termination condition
                        if cv2.waitKey(10) & 0xFF == ord('q'):
                            break
                    vid.release()
                    cv2.destroyAllWindows()
                    testing = False
            else:
                if st.button("Stop the test"):
                    testing = False
                    st.button("Test your knowledge")
                    # Close webcam
                    vid.release()
                    cv2.destroyAllWindows()
        elif category == 'Alphabet':
            st.write('### Alphabet')
            st.write('Here you can learn the Lebanese Sign Language alphabet.')
            # st.image("alphabet.jpg", use_column_width=True)  # Add an image showing alphabet gestures


            from MyModel import MyModel
            from PIL import Image
            import os

            # Load your model
            model = MyModel()

            # Function to overlay an image on the webcam frame
            def putImage(frame, overlay_image, alpha=175):
                base_image = Image.fromarray(frame)
                mask = Image.new('RGBA', overlay_image.size, (0, 0, 0, alpha))
                base_image.paste(overlay_image, mask=mask)
                frame = np.array(base_image)
                return frame

            # Mapping of labels to Arabic text
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

            # Dictionary to store images corresponding to each label
            label2image = {}
            for label in label2text.keys():
                imgname = label + '.jpg'
                imgpath = 'assets/image_lables/' + imgname
                label2image[label] = Image.open(imgpath)

            # Streamlit app
            st.title('Arabic Sign Language Recognition')
            st.write('Please Sign a Letter')
            col1, col2 = st.columns([1, 2])

                        # Display the image "Alphabets.png" in the first column

            st.image("Alphabets.jpg", use_column_width='always',caption='Arabic Alphabets')  # Add an image showing the Arabic alphabet

            # Webcam setup
            cap = cv2.VideoCapture(0)
            stframe = st.empty()
            testing = False

        # Display the webcam feed in the second column

            if not testing:
                if st.button("Test your knowledge"):
                    testing = True


                    # Main loop
                    while testing:
                        ret, frame = cap.read()
                        frame = cv2.flip(frame, 1)

                        # Perform prediction using the model
                        label, prop = model.predict([frame])

                        # Overlay image if prediction confidence is high enough
                        if label[0] in label2image.keys() and prop[0] > 0.5:
                            image = label2image[label[0]]
                            frame = putImage(frame, image)
                        else:
                            cv2.putText(frame, "Please Sign a Letter", (150, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 2, cv2.LINE_AA)

                        # Display the frame
                        stframe.image(frame, channels="BGR")



                    cap.release()
                    cv2.destroyAllWindows()
        elif  category == 'Countries':
            st.write('### Countries')
            st.write('Here you can learn common countries in Lebanese Sign Language.')
            #gesture_videos = [("Lebanon", "videos/Lebanon.mp4"), ("Canada", "videos/Canada.mp4"), ("Europe", "videos/Europe.mp4"), ("France", "videos/France.mp4")]
            #selected_index = st.slider("Select a video", 0, len(gesture_videos) - 1, 0)

            #video_name, video_path = gesture_videos[selected_index]

            #st.write(f"**{video_name}**")
            #st.video(video_path)
            stframe = st.empty()
            testing = False

            if not testing:
                if st.button("Test your knowledge"):
                    testing = True
                    st.button("Stop the test")
                    vid = cv2.VideoCapture(0)
                    while testing:
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
                        actions = np.array(['Canada', 'Europe', 'France', 'Lebanon','no_action'])
                        model = keras.models.load_model('countriesCNN1.h5', compile=False)



                            # Call the start_detection function with st and stframe arguments
                        start_gesture_detection(st,stframe,model,actions)


                        # Break the loop if 'q' is pressed or any other termination condition
                        if cv2.waitKey(10) & 0xFF == ord('q'):
                            break
                    vid.release()
                    cv2.destroyAllWindows()
                    testing = False
            else:
                if st.button("Stop the test"):
                    testing = False
                    st.button("Test your knowledge")
                    # Close webcam
                    vid.release()
                    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
