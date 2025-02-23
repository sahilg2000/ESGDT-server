from flask import Flask, Response, render_template
from flask_socketio import SocketIO, emit
import cv2

app = Flask(__name__)
socketio = SocketIO(app, async_mode="threading", cors_allowed_origins="*")

# Store the last pressed key
last_key = {"value": None}

for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera found at index {i}")
        cap.release()


# Video capture (OBS Virtual Camera)
video_source = cv2.VideoCapture(0)  # Change to tested camera index

@app.route("/")
def index():
    return render_template("index.html") 

def generate_frames():
    while True:
        success, frame = video_source.read()
        if not success:
            break
        else:
            _, buffer = cv2.imencode(".jpg", frame)
            frame = buffer.tobytes()
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@socketio.on('key_press')
def handle_key_press(data):
    """Handle key press events from the client"""
    print(f"Key pressed: {data['key']}")
    emit('update_key', {"key": data['key']}, broadcast=True)
    
if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=8080, allow_unsafe_werkzeug=True)

