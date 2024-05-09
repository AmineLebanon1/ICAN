import base64
import os
from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
import mediapipe as mp
from tensorflow import keras
from playsound import playsound
import random
from flask_socketio import SocketIO, emit
app = Flask(__name__)
socketio = SocketIO(app)

# Initialise detection confidence
lstm_threshold = 0.5
toggle_keypoints = True
mediapipe_detection_confidence = 0.5
import cv2
import numpy as np
from flask import Flask, render_template, send_from_directory
from flask_socketio import SocketIO, emit

app = Flask(__name__, static_folder="./templates/static")
app.config["SECRET_KEY"] = "secret!"
socketio = SocketIO(app)


@app.route("/favicon.ico")
def favicon():
    return send_from_directory(
        os.path.join(app.root_path, "static"),
        "favicon.ico",
        mimetype="image/vnd.microsoft.icon",
    )


def base64_to_image(base64_string):
    # Extract the base64 encoded binary data from the input string
    base64_data = base64_string.split(",")[1]
    # Decode the base64 data to bytes
    image_bytes = base64.b64decode(base64_data)
    # Convert the bytes to numpy array
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    # Decode the numpy array as an image using OpenCV
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return image


@socketio.on("connect")
def test_connect():
    print("Connected")
    emit("my response", {"data": "Connected"})

# Set up Mediapipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
colors = [(245, 221, 173), (245, 185, 265), (146, 235, 193),
          (204, 152, 295), (255, 217, 179), (0, 0, 179)]
# Define functions for Mediapipe detection and drawing landmarks
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False                 
    results = model.process(image)                
    image.flags.writeable = True                   
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

# Define function to extract keypoints
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, lh, rh]) # concatenate all the keypoints that are flattened
   

# Define function for probability visualization
def prob_viz(res, actions, input_frame, colors, threshold):
    output_frame = input_frame.copy()
 
    #print(res)

    multiple = 47

    # num = class index , prob = probability of the class
    for num, prob in enumerate(res):


        
        #print(num, prob)
        if np.argmax(res) == num and  res[np.argmax(res)] >= threshold:

            #print(res[np.argmax(res)])
            (text_width, text_height), baseline = cv2.getTextSize(actions[num]+' '+str(round(prob*100,2))+'% ', cv2.FONT_HERSHEY_SIMPLEX,1, 2)
            
            cv2.rectangle(output_frame, (0,60+num*multiple), (int(prob*text_width), 95+num*multiple), colors[num], -1) #change length of bar depending on probability

            cv2.putText(output_frame, actions[num]+' '+str(round(prob*100,2))+'%', (5, 90+num*multiple), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)
        
        else:
            (text_width, text_height), baseline = cv2.getTextSize(actions[num]+' '+str(round(prob*100,2))+'% ', cv2.FONT_HERSHEY_SIMPLEX,1, 2)

            cv2.rectangle(output_frame, (0,60+num*multiple), (int(prob*text_width), 95+num*multiple), colors[num], -1) #change length of bar depending on probability
            
            cv2.putText(output_frame, actions[num]+' '+str(round(prob*100,2))+'%', (5, 90+num*multiple), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
        

        #thres = 0.5
        #if prob >= thres:
            #cv2.putText(output_frame, actions[num]+' '+str(round(prob*100,2))+'%', (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1, cv2.LINE_AA)

        #cv2.putText(image, text, org, font, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]])
    return output_frame


def overlay_transparent(background, overlay, x, y):
        
        # height and width of background image
        background_width = background.shape[1]
        background_height = background.shape[0]
        
        # if coordinate x and y is larger than background width and height, stop code
        if x >= background_width or y >= background_height:
            return background

        
        
        # height and width of overlay image
        h, w = overlay.shape[0], overlay.shape[1]

        #print('x:',x)
        #print('overlay_width:',w)
        #print('background_width:',background_width)
       

        #print('y:',y)
        #print('overlay_height:',h)
        #print('background_height:',background_width)
        

        if w >= background_width:
            return background
        if h >= background_height:
            return background
        
        # if coordinate x + width of overlay is larger than background width and height, stop code
        if x + w > background_width:
            #w = background_width - x
            #overlay = overlay[:, :w]
            return background
        if x - w < 2:
            #w = background_width - x
            #overlay = overlay[:, :w]
            return background
        if y + h > background_height:
            #h = background_height - y
            #overlay = overlay[:h]
            return background
        
        if y - h < 2:
            #h = background_height - y
            #overlay = overlay[:h]
            return background
        
        if overlay.shape[2] < 4:
            overlay = np.concatenate(
                [
                    overlay,
                    np.ones((overlay.shape[0], overlay.shape[1], 1), dtype = overlay.dtype) * 255
                ],
                axis = 2,
            )

        overlay_image = overlay[..., :3]
        mask = overlay[..., 3:] / 255.0

        background[y:y+h, x:x+w] = (1.0 - mask) * background[y:y+h, x:x+w] + mask * overlay_image

        return background
def add_image(image,results, action):

    #height,width = image.shape
    #print(image.shape)
    width = image.shape[1]#480
    height= image.shape[0]#640

    def overlay_transparent(background, overlay, x, y):
        
        # height and width of background image
        background_width = background.shape[1]
        background_height = background.shape[0]
        
        # if coordinate x and y is larger than background width and height, stop code
        if x >= background_width or y >= background_height:
            return background

        
        
        # height and width of overlay image
        h, w = overlay.shape[0], overlay.shape[1]

        #print('x:',x)
        #print('overlay_width:',w)
        #print('background_width:',background_width)
       

        #print('y:',y)
        #print('overlay_height:',h)
        #print('background_height:',background_width)
        

        if w >= background_width:
            return background
        if h >= background_height:
            return background
        
        # if coordinate x + width of overlay is larger than background width and height, stop code
        if x + w > background_width:
            #w = background_width - x
            #overlay = overlay[:, :w]
            return background
        if x - w < 2:
            #w = background_width - x
            #overlay = overlay[:, :w]
            return background
        if y + h > background_height:
            #h = background_height - y
            #overlay = overlay[:h]
            return background
        
        if y - h < 2:
            #h = background_height - y
            #overlay = overlay[:h]
            return background
        
        if overlay.shape[2] < 4:
            overlay = np.concatenate(
                [
                    overlay,
                    np.ones((overlay.shape[0], overlay.shape[1], 1), dtype = overlay.dtype) * 255
                ],
                axis = 2,
            )

        overlay_image = overlay[..., :3]
        mask = overlay[..., 3:] / 255.0

        background[y:y+h, x:x+w] = (1.0 - mask) * background[y:y+h, x:x+w] + mask * overlay_image

        return background

    index = 10
    
    face_keypoint=np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark])if results.face_landmarks else np.zeros(468*3)
    #print(len(face_keypoint))
    #print(action)
    if face_keypoint.size != 0 and np.any(face_keypoint[index]) == True:

        if action =='Bird':
            file_name = './emoji/bird.png'
        elif action =='Butterfly':
            file_name = './emoji/butterfly.png'
        elif action =='Gorilla':
            file_name = './emoji/gorilla.png'
        elif action == 'Cow':
            file_name = './emoji/cow.png'
        elif action == 'Elephant':
            file_name = './emoji/elephant.png'
        elif action == 'Alligator':
            file_name = './emoji/alligator.png'
        else:
            file_name = './emoji/No_sign.png'
            
        # if action != 'No Action':    
        #     overlay= cv2.imread(file_name, cv2.IMREAD_UNCHANGED)

        #     #overlay= cv2.resize(overlay, (0,0), fx=min(0.1,float(1/face_keypoint[index][2]*-20)), fy=min(0.1,float(1/face_keypoint[index][2]*-20)))
        #     #print('z normalized',face_keypoint[index][2])
        #     #if face_keypoint[index][2]*-100 >1:
        #         #print('close to camera')
        #     #else:
        #         #print('far from camera')

        #     new_z = 0.1/((float(face_keypoint[index][2]*10)-(-1))/(1+1))
        #     #print('new_z',new_z)
        #     #print('z ',face_keypoint[index][2]*-10)
        #     #print('fx:',new_z)
        #     #print('fy:',new_z)

        #     #print(min(0.5,float(new_z)))

        #     overlay= cv2.resize(overlay, (0,0), fx=min(0.5,abs(float(new_z))), fy=min(0.5,abs(float(new_z))))

        #     #print('Normalized',face_keypoint[index])
        #     x = int(float(face_keypoint[index][0])*width)
        #     y = int(float(face_keypoint[index][1])*height)
        #     #print('Actual x',x)
        #     #print('Actual y',y)
        #     #cv2.circle(image,(x,y),3,(255,255,0),thickness= -1)

        #     #overlay = img2.copy()
        #     #image = cv2.rectangle(image, (x,y), (x+overlay.shape[1],y-overlay.shape[0]), (255,0,0), 3)

        #     #image = cv2.addWeighted(image,0.4,overlay,0.1,0)

        #     image = overlay_transparent(image, overlay, x - int(overlay.shape[0]/2), y-overlay.shape[0])


        #     #Setting the paste destination coordinates. For the time being, in the upper left
        #     #x1, y1, x2, y2 = x, y, overlay.shape[1], overlay.shape[0]

        #     #Synthetic!
        #     #image[y1:y2, x1:x2] = overlay[y1:y2, x1:x2]

    
# define extract keypoint function
def extract_keypoints(results):
    key1 = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    key2 = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([key1, key2, lh, rh])

def draw_landmarks(image, results): # draw landmarks for each image/frame
    
    
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections

def draw_styled_landmarks(image, results): # draw landmarks for each image/frame, fix colour of landmark drawn
    
    # Draw face connections
    
    # left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    # right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             ) 

# Load the model
model = keras.models.load_model('rec_11.h5',compile=False)

# Define actions
actions = np.array(['Bird', 'Bee', 'Rabbit', 'Snake', 'Fish', 'Cat'])

@socketio.on('generate new action')
def emit_new_action():
    random_action()
    socketio.emit('new action', {'data': current_action})



@app.route("/get_current_action", methods=['GET'])
def get_current_action():
    return jsonify(current_action)

current_score = 0
sentence = []

@app.route("/get_next_action", methods=['GET'])
def get_next_action():
    current_action = random_action()
    return jsonify(current_action)


def random_action():
    global current_action
    newAction = random.choice(actions_list)

    # while the new action is equal to the previous action, choose a new action
    while newAction == current_action:
        newAction = random.choice(actions_list)

    current_action = newAction
    #print('Current Action:', current_action)
    return current_action
@app.route("/get_current_score", methods=['GET'])
def get_current_score():
    return jsonify(current_score)

reset_score_frame_count = 0

@app.route('/reset_score')
def reset_score():
    global current_score
    global reset_score_frame_count
    global sentence
    current_score = 0
    sentence = []

    reset_score_frame_count = 10
    print('current_score', current_score)

    return("nothing")
sequence = []
@socketio.on("image")
def receive_image(image):
    # Decode the base64-encoded image data
    image = base64_to_image(image)
    predictions = []
    global sequence
    global frame_count
    frame_count = 0
    global current_score
    global reset_score_frame_count
    global lstm_threshold
    global sentence
    global toggle_keypoints

    width = image.shape[1]  # Define width and height here
    height = image.shape[0]

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        # Perform holistic detection
        image, results = mediapipe_detection(image, holistic)
        keypoints = extract_keypoints(results)

        # Append keypoints to sequence
        sequence.append(keypoints)
        sequence = sequence[-30:]

        # Loading model screen if sequence is not complete
        if len(sequence) < 30:
            width = image.shape[1]
            height = image.shape[0]
            alpha = 0.5

            overlay = image.copy()
            cv2.rectangle(overlay, (0, 0), (width, height), (255, 255, 255), -1)
            cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
            cv2.putText(image, 'Loading...', (width//2 - 100, height//2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

        # Draw landmarks when lstm model is ready
        if toggle_keypoints and len(sequence) == 30:
            draw_styled_landmarks(image, results)
        print('frame_count:', frame_count)
        # Make predictions if frame_count reaches 30
        if len(sequence) == 30:
            frame_count += 1  # Move this line after the check
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            predictions.append(np.argmax(res))

            # Reset frame_count
            frame_count = 0

            # Vizualization logic
            if np.unique(predictions[-10:])[0] == np.argmax(res):
                if res[np.argmax(res)] >= lstm_threshold and actions[np.argmax(res)] == current_action and frame_count == 0:
                    print('Correct!')
                    frame_count = 15
                    emit_new_action()
                    current_score += 1

                    if len(sentence) > 0:
                        if actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                    else:
                        sentence.append(actions[np.argmax(res)])

                if res[np.argmax(res)] >= lstm_threshold and actions[np.argmax(res)] != 'No Action':
                    add_image(image, results, str(actions[np.argmax(res)]))

            if len(sentence) > 15:
                sentence = sentence[-15:]

            image = prob_viz(res, actions, image, colors, lstm_threshold)

        # Display correct classes top display bar
        cv2.rectangle(image, (0, 0), (width, 50), (0, 60, 123), -1)
        if len(sentence) > 0:
            cv2.putText(image, ' ' + sentence[-1], (3, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (245, 221, 173), 2, cv2.LINE_AA)
            (text_width, text_height), baseline = cv2.getTextSize(sentence[-1], cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        if len(sentence) > 1:
            cv2.putText(image, '  ' + ' '.join(sentence[::-1][1:]), (text_width, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        if frame_count > 0:
            # display_correct_screen(image)
            width = image.shape[1]  # 480
            height = image.shape[0]  # 640
            alpha = 0.5

            overlay = image.copy()

            cv2.rectangle(overlay, (0, 0), (width, height),
                            (144, 250, 144), -1)

            # apply the overlay
            cv2.addWeighted(overlay, alpha, image, 1 - alpha,
                            0, image)

            cv2.putText(image, 'CORRECT!', (width//2 - 75, height//2 + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            overlay = cv2.imread(
                './static/images/correct1.png', cv2.IMREAD_UNCHANGED)
            image = overlay_transparent(
                image, overlay, width//2 - 35, height//2-70)

            frame_count -= 1
        # Display reset score screen
        if reset_score_frame_count > 0:
            width = image.shape[1]
            height = image.shape[0]
            alpha = 0.5

            overlay = image.copy()
            cv2.rectangle(overlay, (0, 0), (width, height), (255, 255, 255), -1)
            cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

            (text_width, text_height), baseline = cv2.getTextSize('Score Reset!', cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            cv2.putText(image, 'Score Reset!', (width//2 - text_width//2, height//2 + text_height), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

            reset_score_frame_count -= 1

        # Encode output image to bytes and emit processed image
        frame = cv2.imencode('.jpg', image)[1].tobytes()
        processed_img_data = b"data:image/jpeg;base64," + base64.b64encode(frame)
        emit("processed_image", processed_img_data.decode())


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html', current_action=current_action, current_score=current_score)

actions_list = list(['Bird', 'Bee', 'Rabbit', 'Snake', 'Fish', 'Cat'])
current_action = random.choice(actions_list)

#if __name__ == "__main__":
#    socketio.run(app, host='127.0.0.1', port=5000, debug=True)

if __name__ == '__main__':
    socketio.run(app, cors_allowed_origins='*')
