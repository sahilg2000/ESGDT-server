from flask import Flask, Response, render_template
from flask_socketio import SocketIO, emit
import cv2
import numpy as np

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
video_source = cv2.VideoCapture(1) # Change to tested camera index

@app.route("/")
def index():
    return render_template("index.html") 

def generate_frames():
    while True:
        success, frame = video_source.read()
        if not success:
            break
        
        # Convert to HLS for better control over brightness and color
        hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)

        # Color threshold to isolate "white" lines
        lower_white = np.array([0, 200, 0], dtype=np.uint8) 
        upper_white = np.array([179, 255, 60], dtype=np.uint8)
        mask_white = cv2.inRange(hls, lower_white, upper_white)

        # Isolate the white lines using morphological operations
        kernel = np.ones((5, 5), np.uint8)
        mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_OPEN, kernel)
        mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_CLOSE, kernel)

        # Uncomment to visualize the mask and 
        # change 'ret, buffer = cv2.imencode('.jpg', frame)' to 'ret, buffer = cv2.imencode('.png', mask_vis)'
        #mask_vis = cv2.cvtColor(mask_white, cv2.COLOR_GRAY2BGR)
        
        # Edge detection on the color mask
        edges = cv2.Canny(mask_white, 50, 150)

        # Hough Transform to detect line segments
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50,
                                minLineLength=50, maxLineGap=150)

        # Draw the detected lines on the original frame
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        ret, buffer = cv2.imencode('.jpg', frame)
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

