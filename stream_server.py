from flask import Flask, Response, render_template
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import win32gui
import win32ui
import win32con

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Store the last pressed key
last_key = {"value": None}

def get_game_window():
    """Find the Bevy game window"""
    def callback(hwnd, windows):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if "car_demo" in title.lower():  # Change this to match your window title
                windows.append(hwnd)
    windows = []
    win32gui.EnumWindows(callback, windows)
    return windows[0] if windows else None

def capture_game_window():
    """Capture the game window content"""
    try:
        hwnd = get_game_window()
        if not hwnd:
            print("Game window not found")
            return None

        left, top, right, bottom = win32gui.GetWindowRect(hwnd)
        width = right - left
        height = bottom - top

        hwndDC = win32gui.GetWindowDC(hwnd)
        mfcDC = win32ui.CreateDCFromHandle(hwndDC)
        saveDC = mfcDC.CreateCompatibleDC()
        saveBitMap = win32ui.CreateBitmap()
        saveBitMap.CreateCompatibleBitmap(mfcDC, width, height)
        saveDC.SelectObject(saveBitMap)
        saveDC.BitBlt((0, 0), (width, height), mfcDC, (0, 0), win32con.SRCCOPY)

        bmpstr = saveBitMap.GetBitmapBits(True)
        img = np.frombuffer(bmpstr, dtype=np.uint8)
        img.shape = (height, width, 4)

        # Cleanup
        win32gui.DeleteObject(saveBitMap.GetHandle())
        saveDC.DeleteDC()
        mfcDC.DeleteDC()
        win32gui.ReleaseDC(hwnd, hwndDC)

        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img

    except Exception as e:
        print(f"Error capturing window: {e}")
        return None

for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera found at index {i}")
        cap.release()


# Video capture (OBS Virtual Camera)
video_source = cv2.VideoCapture(1)  # Change to tested camera index

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
    socketio.run(app, host="0.0.0.0", port=8080)
