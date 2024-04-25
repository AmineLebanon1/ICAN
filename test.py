from flask import Flask, render_template, Response, jsonify, session
import cv2
import numpy as np
from MyModel import MyModel
from PIL import Image
import random
import os
import secrets

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

webcam_on = False  # Flag to track if the webcam feed is on
model = MyModel()
signs_dir = 'image_lables/'
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

labels = list(label2text.keys())

def get_random_sign():
    sign_label = random.choice(labels)
    sign_image_path = os.path.join(signs_dir, sign_label + '.png')
    return sign_image_path

label2image = {}
for label in labels:
    imgname = label + '.png'
    imgpath = 'static/image_lables/' + imgname
    label2image[label] = Image.open(imgpath)

def putImage(frame, overlay_image, alpha=175):
    base_image = Image.fromarray(frame)
    mask = Image.new('RGBA', overlay_image.size, (0, 0, 0, alpha))
    base_image.paste(overlay_image, mask=mask)
    frame = np.array(base_image)
    return frame

def get_random_sign_name():
    if 'current_sign_name' not in session:
        session['current_sign_name'] = random.choice(labels)
    return session['current_sign_name']

def generate_frames(current_sign_name):
    with app.test_request_context("/"):

        points = session.get('points', 0)
        cap = cv2.VideoCapture(0)
        sign_detected = False  
        while True:
            success, frame = cap.read()
            if not success:
                break
            else:
                frame = cv2.flip(frame, 1)
                label, prop = model.predict([frame])
                print("Predicted sign:", label[0]) 
            
                # Check if the predicted label matches any of the sign labels
                if label[0] in labels:
                    if label[0] == current_sign_name:
                        sign_detected = True
                        points += 1
                        print("New sign name:", current_sign_name)
                        # Display message on the frame
                        cv2.putText(frame, "CORRECT! You Gained One Point", (100, 250),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                # Reset sign_detected flag if no sign is detected
                if label[0] == '-1' or prop[0] <= 0.5:
                    sign_detected = False
                
                # Render the sign image if detected
                if label[0] in label2image:
                    image = label2image[label[0]]
                    frame = putImage(frame, image)
                
                cv2.putText(frame, "Points: " + str(points), (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
   session['current_sign_name'] = random.choice(labels)
   current_sign_name = session['current_sign_name']
   return render_template('test.html', current_sign_name=label2text[current_sign_name])


@app.route('/update_data', methods=['POST'])
def update_data():
    # Get the current sign name and points from session or wherever they are stored
    current_sign_name = session['current_sign_name']
    points = session.get('points', 0)
    # Return the current sign name and points as JSON response
    return jsonify(current_sign_name=current_sign_name, points=points)


@app.route('/video_feed')
def video_feed():
    current_sign_name = session['current_sign_name']
    return Response(generate_frames(current_sign_name), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=8080)
