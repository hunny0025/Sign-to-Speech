# Install dependencies:
# pip install mediapipe opencv-python pyttsx3

import cv2
import mediapipe as mp
import pyttsx3

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1)
engine = pyttsx3.init()

# Finger indices in MediaPipe
FINGER_TIPS = [4, 8, 12, 16, 20]

# Define gestures based on finger positions (up or down)
def fingers_up(hand_landmarks):
    tips = [hand_landmarks.landmark[i] for i in FINGER_TIPS]
    fingers = []
    # Thumb
    if tips[0].x < hand_landmarks.landmark[2].x:
        fingers.append(1)
    else:
        fingers.append(0)
    # Other fingers
    for i in range(1,5):
        if tips[i].y < hand_landmarks.landmark[i*4-2].y:
            fingers.append(1)
        else:
            fingers.append(0)
    return fingers  # list of 0 (down) or 1 (up)

# Map finger patterns to words (20+ gestures)
GESTURES = {
    (0,1,0,0,0): "Hello",
    (1,1,1,1,1): "Yes",
    (0,0,0,0,0): "No",
    (1,0,0,0,0): "Thank you",
    (0,1,1,0,0): "Please",
    (0,1,0,0,1): "Sorry",
    (1,1,0,0,0): "Good",
    (0,1,1,1,0): "Bad",
    (0,1,1,1,1): "Love",
    (1,0,1,0,1): "Help",
    (1,0,0,1,0): "Stop",
    (0,0,1,0,0): "Yes Sir",
    (0,1,0,1,0): "No Sir",
    (1,0,1,1,0): "Friend",
    (1,1,0,1,1): "Family",
    (1,0,1,1,1): "Food",
    (0,1,0,0,1): "Water",
    (1,0,0,0,1): "Home",
    (0,1,0,1,1): "School",
    (1,1,1,0,0): "Work",
    (0,0,1,1,0): "Go",
    (1,1,0,0,1): "Come"
}

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    gesture_text = ""
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
            fingers = tuple(fingers_up(handLms))
            gesture_text = GESTURES.get(fingers, "")
            if gesture_text:
                engine.say(gesture_text)
                engine.runAndWait()

    if gesture_text:
        cv2.putText(frame, gesture_text, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow("Sign to Speech", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
