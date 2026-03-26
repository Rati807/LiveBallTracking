import cv2
import numpy as np

# Start camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera open nahi ho raha")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame nahi mil raha")
        break

    # Blur for smooth detection
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)

    # Convert BGR to HSV
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Red color range (adjust if ball ka color different ho)
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red, upper_red)

    # Upper red hue range
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    mask = mask1 + mask2

    # Remove noise
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt) > 500:
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            if radius > 10:
                center = (int(x), int(y))
                radius = int(radius)
                # Draw circle
                cv2.circle(frame, center, radius, (0, 255, 0), 2)
                cv2.putText(frame, "Ball", (center[0]-20, center[1]-20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show result
    cv2.imshow("Ball Tracker", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()