# app.py
from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
import numpy as np
import pickle
import json

app = Flask(__name__)

# Load the trained model
model_dict = pickle.load(open('C:\\Users\\Lenovo - LOQ\\Desktop\\pattern project\\sign\\model.p', 'rb'))
model = model_dict['model']

# Define labels dictionary for ASL letters
labels_dict = { 
    0: "A",
    1: "B",
    2: "G",
    3: "L",
    4: "S",
    5: "Space",
    6: "nothing",
    7: "Z",
    8: "Y",
    9: "W",
    10 : "O",
    11 : "P",
    12 : "N",
    13 : "J",
   }

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize the hands object
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def draw_colored_landmarks(frame, hand_landmarks, theme_color):
    finger_colors = {
        'THUMB': (255, 0, 0),
        'INDEX': (0, 255, 0),
        'MIDDLE': (0, 0, 255),
        'RING': (255, 0, 255),
        'PINKY': (0, 255, 255)
    }
    
    finger_landmarks = {
        'THUMB': [1, 2, 3, 4],
        'INDEX': [5, 6, 7, 8],
        'MIDDLE': [9, 10, 11, 12],
        'RING': [13, 14, 15, 16],
        'PINKY': [17, 18, 19, 20]
    }
    
    palm_landmarks = [0]
    h, w = frame.shape[:2]
    
    for idx in palm_landmarks:
        pos = hand_landmarks.landmark[idx]
        cx, cy = int(pos.x * w), int(pos.y * h)
        cv2.circle(frame, (cx, cy), 5, theme_color, -1)
    
    for finger, indices in finger_landmarks.items():
        color = finger_colors[finger]
        for idx in indices:
            pos = hand_landmarks.landmark[idx]
            cx, cy = int(pos.x * w), int(pos.y * h)
            cv2.circle(frame, (cx, cy), 5, color, -1)
            if idx > 0:
                prev_pos = hand_landmarks.landmark[idx - 1]
                prev_cx, prev_cy = int(prev_pos.x * w), int(prev_pos.y * h)
                cv2.line(frame, (prev_cx, prev_cy), (cx, cy), color, 2)

def generate_frames():
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        predicted_character = None
        
        if results.multi_hand_landmarks:
            data_aux = []
            x_ = []
            y_ = []
            
            for hand_landmarks in results.multi_hand_landmarks:
                draw_colored_landmarks(frame, hand_landmarks, (200, 180, 255))
                
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)
                
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))
            
            try:
                data_aux = np.asarray(data_aux).reshape(1, -1)
                prediction = model.predict(data_aux)
                predicted_character = labels_dict.get(int(prediction[0]), '?')
                
                # Draw bounding box
                H, W = frame.shape[:2]
                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10
                x2 = int(max(x_) * W) + 10
                y2 = int(max(y_) * H) + 10
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 180, 255), 2)
                
                # Corner accents
                corner_length = 20
                cv2.line(frame, (x1, y1), (x1 + corner_length, y1), (200, 180, 255), 3)
                cv2.line(frame, (x1, y1), (x1, y1 + corner_length), (200, 180, 255), 3)
                cv2.line(frame, (x2, y2), (x2 - corner_length, y2), (200, 180, 255), 3)
                cv2.line(frame, (x2, y2), (x2, y2 - corner_length), (200, 180, 255), 3)
                
                # Add prediction text
                cv2.putText(frame, predicted_character, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.3, (200, 180, 255), 3)
                
            except Exception as e:
                print(f"Error in prediction: {e}")
        
        # Convert frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        # Send the frame and prediction
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)