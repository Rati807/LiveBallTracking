from flask import Flask, render_template, Response
import cv2
import numpy as np

app = Flask(__name__)

cap = cv2.VideoCapture(0)

def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        lower_red = np.array([0, 120, 70])
        upper_red = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red, upper_red)

        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

        mask = mask1 + mask2

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        status = "LOST ❌"

        for cnt in contours:
            if cv2.contourArea(cnt) > 500:
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                if radius > 10:
                    center = (int(x), int(y))
                    radius = int(radius)

                    status = "FOUND ✅"

                    cv2.circle(frame, center, radius, (0, 255, 0), 2)

                    cv2.putText(frame, f"X:{center[0]} Y:{center[1]}",
                                (center[0]-40, center[1]-40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        cv2.putText(frame, f"Ball Status: {status}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    print("Server start ho raha hai...")
    app.run(debug=True)