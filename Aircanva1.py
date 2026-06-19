import cv2
import numpy as np
import mediapipe as mp
from collections import deque

# Define color arrays
bpoints = [deque(maxlen=1024)]
gpoints = [deque(maxlen=1024)]
rpoints = [deque(maxlen=1024)]
ypoints = [deque(maxlen=1024)]

# Indexes to track each new stroke
blue_index = green_index = red_index = yellow_index = 0

# Define drawing colors (BGR)
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]  # Blue, Green, Red, Yellow
colorIndex = -1  # No color selected initially

# Canvas for painting
paintWindow = np.ones((471, 636, 3), dtype=np.uint8) * 255

# Draw UI buttons
buttons = [((40, 1), (140, 65), "CLEAR", (0, 0, 0)),
           ((160, 1), (255, 65), "BLUE", (255, 0, 0)),
           ((275, 1), (370, 65), "GREEN", (0, 255, 0)),
           ((390, 1), (485, 65), "RED", (0, 0, 255)),
           ((505, 1), (600, 65), "YELLOW", (0, 255, 255))]

for (start, end, text, color) in buttons:
    cv2.rectangle(paintWindow, start, end, color, 2)
    cv2.putText(paintWindow, text, (start[0]+10, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2, cv2.LINE_AA)

cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE)

# Initialize MediaPipe Hands
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Draw UI on the frame
    for (start, end, text, color) in buttons:
        cv2.rectangle(frame, start, end, color, 2)
        cv2.putText(frame, text, (start[0]+10, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2, cv2.LINE_AA)

    # Detect hand landmarks
    result = hands.process(framergb)

    if result.multi_hand_landmarks:
        for handslms in result.multi_hand_landmarks:
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

            # Extract landmarks
            landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in handslms.landmark]
            fore_finger = landmarks[8]
            thumb = landmarks[4]
            center = fore_finger

            cv2.circle(frame, center, 4, (0, 255, 0), -1)

            # Gesture: Check if thumb is close to index finger (new stroke)
            distance = np.linalg.norm(np.array(fore_finger) - np.array(thumb))
            if distance < 30:
                bpoints.append(deque(maxlen=1024)); blue_index += 1
                gpoints.append(deque(maxlen=1024)); green_index += 1
                rpoints.append(deque(maxlen=1024)); red_index += 1
                ypoints.append(deque(maxlen=1024)); yellow_index += 1

            # Button Press Detection (top of screen)
            elif center[1] <= 65:
                if 40 <= center[0] <= 140:
                    # Clear canvas
                    bpoints = [deque(maxlen=1024)]; gpoints = [deque(maxlen=1024)]
                    rpoints = [deque(maxlen=1024)]; ypoints = [deque(maxlen=1024)]
                    blue_index = green_index = red_index = yellow_index = 0
                    paintWindow[67:, :, :] = 255
                elif 160 <= center[0] <= 255:
                    colorIndex = 0
                elif 275 <= center[0] <= 370:
                    colorIndex = 1
                elif 390 <= center[0] <= 485:
                    colorIndex = 2
                elif 505 <= center[0] <= 600:
                    colorIndex = 3
            else:
                # Draw according to selected color
                if colorIndex == 0:
                    bpoints[blue_index].appendleft(center)
                elif colorIndex == 1:
                    gpoints[green_index].appendleft(center)
                elif colorIndex == 2:
                    rpoints[red_index].appendleft(center)
                elif colorIndex == 3:
                    ypoints[yellow_index].appendleft(center)
    else:
        # No hand detected: prepare for next stroke
        bpoints.append(deque(maxlen=1024)); blue_index += 1
        gpoints.append(deque(maxlen=1024)); green_index += 1
        rpoints.append(deque(maxlen=1024)); red_index += 1
        ypoints.append(deque(maxlen=1024)); yellow_index += 1

    # Draw lines on canvas and frame
    points = [bpoints, gpoints, rpoints, ypoints]
    for i in range(len(points)):
        for j in range(len(points[i])):
            for k in range(1, len(points[i][j])):
                if points[i][j][k - 1] is None or points[i][j][k] is None:
                    continue
                cv2.line(frame, points[i][j][k - 1], points[i][j][k], colors[i], 2)
                cv2.line(paintWindow, points[i][j][k - 1], points[i][j][k], colors[i], 2)

    # Display output
    cv2.imshow("Output", frame)
    cv2.imshow("Paint", paintWindow)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        cv2.imwrite("MyDrawing.png", paintWindow)
        print("Drawing saved as MyDrawing.png")

# Cleanup
cap.release()
cv2.destroyAllWindows()
