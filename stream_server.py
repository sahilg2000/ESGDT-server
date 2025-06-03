import cv2
import numpy as np
import time
import asyncio
import websockets
import threading

from flask import Flask, Response, render_template
from flask_socketio import SocketIO, emit

app = Flask(__name__)
socketio = SocketIO(app, async_mode="threading", cors_allowed_origins="*")


# Global dictionary storing the current control values
CONTROL = {
    "throttle": 0.0,   # Slow forward by default 
    "brake": 0.0,
    "steering": 0.0
}

for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera found at index {i}")
        cap.release()

SEND_CONTROL = False          # start without sending control commands
VIEW_MASK    = False          # start in normal view
BRAKE_OVERRIDE_UNTIL = 0.0  

# Video capture (OBS Virtual Camera)
video_source = cv2.VideoCapture(1) # Change to tested camera index

@app.route("/")
def index():
    return render_template("index.html")

#  classify a single Hough segment by voting on the three masks
def classify_line_with_masks(x1, y1, x2, y2, masks):
    """
    Decide whether a line segment is cyan, magenta or white by counting
    how many pixels of that segment fall inside each binary mask.
    """
    h, w = next(iter(masks.values())).shape
    tmp = np.zeros((h, w), dtype=np.uint8)
    cv2.line(tmp, (x1, y1), (x2, y2), 255, 3)

    votes = {name: cv2.countNonZero(cv2.bitwise_and(tmp, m))
             for name, m in masks.items()}

    # choose the mask with most overlap
    label = max(votes, key=votes.get)

    outline_bgr = {                 # use contrasting colours
        "cyan":    (  0,165,255),   # orange
        "magenta": (  0,255,  0),   # lime-green
        "white":   (255,  0,  0),   # blue
    }[label]

    return label, outline_bgr

def generate_frames():

    while True:
        success, frame = video_source.read()
        if not success:
            break

        now = time.time()
        
        # Convert to HLS for better control over brightness and color
        frame_hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)

        # Color threshold to isolate color lines
        mask_white = cv2.inRange(frame_hls,
                                 np.array([  0,200,   0], dtype=np.uint8),
                                 np.array([179,255,  70], dtype=np.uint8))

        mask_cyan = cv2.inRange(frame_hls,
                                np.array([ 80, 50, 100], dtype=np.uint8),
                                np.array([105,255,255], dtype=np.uint8))

        mask_mag = cv2.inRange(frame_hls,
                               np.array([135, 50, 100], dtype=np.uint8),
                               np.array([170,255,255], dtype=np.uint8))

        # Morphological clean-up on every mask
        kernel = np.ones((5, 5), np.uint8)
        for m in (mask_white, mask_cyan, mask_mag):
            m[:] = cv2.morphologyEx(m, cv2.MORPH_OPEN,  kernel)
            m[:] = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)

        # Edge detection on the combined colour mask
        mask_all = cv2.bitwise_or(mask_white,
                                  cv2.bitwise_or(mask_cyan, mask_mag))
        edges = cv2.Canny(mask_all, 50, 150)

        # Hough Transform to detect line segments
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                                threshold=50,
                                minLineLength=50,
                                maxLineGap=150)

        detected_colours = set()

         # Determine the colour of every line segment by mask voting
        if lines is not None:
            masks = {"cyan": mask_cyan, "magenta": mask_mag, "white": mask_white}
            for (x1, y1, x2, y2) in lines[:, 0]:
                label, outline = classify_line_with_masks(x1, y1, x2, y2, masks)
                detected_colours.add(label)
                cv2.line(frame, (x1, y1), (x2, y2), outline, 3)

        if now < BRAKE_OVERRIDE_UNTIL:
            CONTROL.update(throttle=0.0, brake=1.0)
        else:
            if "magenta" in detected_colours:
                CONTROL["throttle"] = 0.0   
                CONTROL["brake"]    =   0.2 # slow down
            elif "cyan" in detected_colours:
                CONTROL["throttle"] = 0.15   # Constant speed
                CONTROL["brake"]    = 0.15   
            else:                           
                CONTROL["throttle"] = 0.15   # Accelerate
                CONTROL["brake"]    = 0.0
            
        # Compute a simple "steering" from the lines by finding the lane center at the bottom row of the image.
        steering = compute_steering_from_lines(lines, frame.shape)
        CONTROL["steering"] = steering

        # FPS calculation using moving average
        now = time.time()
        if not hasattr(generate_frames, "last_time"):
            generate_frames.last_time = now
            fps = 0.0
        else:
            fps = 1.0 / max(1e-6, now - generate_frames.last_time)
            generate_frames.last_time = now

        # Count the number of lines detected
        line_count = len(lines) if lines is not None else 0

        # Mask coverage % = non-zero edge pixels in ROI
        edge_pixels = np.count_nonzero(edges)
        total_pixels = edges.shape[0] * edges.shape[1]
        mask_coverage = 100.0 * edge_pixels / total_pixels

        # Raw offset calculation for telemetry (even if not used in steering)
        center_x = frame.shape[1] / 2.0
        avg_x = sum([(x1 + x2) / 2.0 for [[x1, _, x2, _]] in lines]) / len(lines) if lines is not None else center_x
        offset_px = center_x - avg_x

        # Approximate steering angle using FOV assumption
        fov_deg = 60.0  # assumed camera field of view in degrees
        steering_angle_deg = CONTROL["steering"] * (fov_deg / 2)

        # Emit telemetry data to frontend via Socket.IO
        socketio.emit("telemetry", {
            "offset_px": offset_px,
            "steering": CONTROL["steering"],
            "steering_angle": steering_angle_deg,
            "throttle": CONTROL["throttle"],
            "brake": CONTROL["brake"],
            "fps": fps,
            "line_count": line_count,
            "mask_coverage": mask_coverage
        })

        # Encode and yield the frame
        output_frame = (cv2.cvtColor(mask_all, cv2.COLOR_GRAY2BGR)
                        if VIEW_MASK else frame)
        ret, buffer = cv2.imencode('.jpg', output_frame)
        if not ret:
            break
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")

# Compute a simple steering angle in [-1..1] based on detected line midpoints.
def compute_steering_from_lines(lines, shape):
    
    if lines is None or len(lines) == 0:
        return 0.0  # if no lines detected, drive straight


    # Average the midpoint of each line if the line crosses the bottom region.
    height, width, _ = shape
    bottom_region_y = int(height * 0.8) # define bottom region at 80% of frame height

    xs = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # check if either endpoint is near bottom
        if y1 > bottom_region_y or y2 > bottom_region_y:
            # take midpoint
            xm = (x1 + x2) / 2.0
            xs.append(xm)

    if len(xs) == 0:
        return 0.0  # If no lines near bottom, drive straight

    # Compute the average x position of the lines
    avg_x = sum(xs) / len(xs)
    center_x = width / 2.0
    offset = center_x - avg_x  # If positive, the lane center is left so steer left.

    # map offset to [-1..1] with a simple proportional gain
    Kp = 0.003
    steering = Kp * offset

    # clamp the steering to -1..1
    steering = max(-1.0, min(1.0, steering))
    return steering


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@socketio.on('toggle_view')
def handle_toggle_view():
    global VIEW_MASK
    VIEW_MASK = not VIEW_MASK
    print(f"View mode: {'MASK' if VIEW_MASK else 'NORMAL'}")
    

@socketio.on('toggle_control')
def handle_toggle_control():
    global SEND_CONTROL, BRAKE_OVERRIDE_UNTIL

    if SEND_CONTROL:                          # pause external control
        BRAKE_OVERRIDE_UNTIL = time.time() + 1.5   # 1.5s hard-brake window
        SEND_CONTROL = False                       
        emit('control_status', {
            "message": "Control pausing: applying brakes",
            "enabled": False
        }, broadcast=True)

    else:                                      # resume external control
        SEND_CONTROL = True
        BRAKE_OVERRIDE_UNTIL = 0.0             # reset brake override
        CONTROL.update(throttle=0.1, brake=0.0)
        print("[CONTROL] Control resumed.")
        emit('control_status', {
            "message": "Control resumed.",
            "enabled": True
        }, broadcast=True)

async def ws_sender_loop():
    """Asynchronously connect to the simulation's WebSocket and send throttle/brake/steering."""
    uri = "ws://127.0.0.1:8080"
    global SEND_CONTROL
    while True:
        try:
            print("Attempting to connect to the sim's WebSocket on 8080...")
            async with websockets.connect(uri) as websocket:
                print("Connected to sim's WebSocket!")
                while True:
                    # if SEND_CONTROL is True, send the control commands
                    if SEND_CONTROL or time.time() < BRAKE_OVERRIDE_UNTIL:
                        t = CONTROL["throttle"]
                        b = CONTROL["brake"]
                        s = CONTROL["steering"]
                        msg = f"{t:.2f} {b:.2f} {s:.2f}"
                        await websocket.send(msg)
                    await asyncio.sleep(0.05) # ~20 updates/sec
        except Exception as e:
            # if something fails, wait 2s and retry
            print(f"[WS] Error: {e}. Reconnecting in 2s...")
            await asyncio.sleep(2)

def start_ws_loop_in_bg():
    """Start the ws_sender_loop() in a separate thread & event loop."""
    def run_loop():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(ws_sender_loop())

    thr = threading.Thread(target=run_loop, daemon=True)
    thr.start()

if __name__ == "__main__":
    # launch the WebSocket sending loop in the background
    start_ws_loop_in_bg()
    socketio.run(app, host="0.0.0.0", port=8080, allow_unsafe_werkzeug=True)