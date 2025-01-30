from flask import Flask, render_template, Response, jsonify
import cv2
import pyaudio
import threading
import time
import requests
import mediapipe as mp
from playsound import playsound

# Import sound and emotion detection module
import sound_emotion_detection_

app = Flask(__name__)

# Initialize the webcam
camera = cv2.VideoCapture(0)

# Global variable to store sound and emotion detection results
detection_results = {"emotion": "neutral", "sound": "normal"}

# Audio streaming configuration
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
audio = pyaudio.PyAudio()

# SOS detection variable
sos_detected = False  # To track the current SOS detection status

# Initialize MediaPipe hands solution
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# Helper function to check if fingers are closed
def fingers_closed(landmarks):
    thumb = [1, 2, 3, 4]
    index_finger = [5, 6, 7, 8]
    middle_finger = [9, 10, 11, 12]
    ring_finger = [13, 14, 15, 16]
    pinky = [17, 18, 19, 20]

    thumb_closed = landmarks[thumb[3]].x < landmarks[thumb[0]].x
    index_closed = landmarks[index_finger[3]].y > landmarks[index_finger[0]].y
    middle_closed = landmarks[middle_finger[3]].y > landmarks[middle_finger[0]].y
    ring_closed = landmarks[ring_finger[3]].y > landmarks[ring_finger[0]].y
    pinky_closed = landmarks[pinky[3]].y > landmarks[pinky[0]].y

    return thumb_closed and index_closed and middle_closed and ring_closed and pinky_closed

# Function to detect SOS gesture
def detect_sos(frame):
    global sos_detected
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            if fingers_closed(hand_landmarks.landmark):
                sos_detected = True
                # Play a sound when SOS is detected
                playsound("C:/Users/manis/Music/race-start-beeps-125125.mp3")
            else:
                sos_detected = False
    else:
        sos_detected = False

def generate_frames():
    """Generator function to stream video frames."""
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Detect SOS gesture
            detect_sos(frame)
            
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Yield frame as a byte array
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    # Main route to show the video and audio streams
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    # Video streaming route
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/sos_status', methods=['GET'])
def sos_status():
    # Endpoint to check if SOS is detected
    return jsonify({'sos_detected': sos_detected})

@app.route('/emotion_detection', methods=['GET'])
def emotion_detection():
    # Dummy detection logic; replace with actual sound and emotion detection logic
    detection_results = sound_emotion_detection_.detect()
    return jsonify(detection_results)

@app.route('/get_info', methods=['GET'])
def get_info():
    # Placeholder for CCTV info, replace this with actual info fetching logic
    ip = requests.get('https://api.ipify.org').text
    location = "SRM Tech Park, Potheri, Chengalpattu, India"  # You can dynamically fetch this as needed
    return jsonify({'ip': ip, 'location': location})

    
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
