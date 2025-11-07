import cv2
import mediapipe as mp
import time
import streamlit as st
from gtts import gTTS
import os
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# ---------------------
# Speech function
# ---------------------
def speak_text(text):
    if text:
        tts = gTTS(text=text, lang='en')
        tts.save("speech.mp3")
        audio_file = open("speech.mp3", "rb")
        st.audio(audio_file, format="audio/mp3")
        os.remove("speech.mp3")

# ---------------------
# Gesture setup
# ---------------------
FINGER_TIPS = [4, 8, 12, 16, 20]
MAX_WORDS = 4
sentence = []
last_word = ""
last_time = time.time()
delay = 1

# Gesture mapping (20+ signs)
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

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1)

# ---------------------
# Detect fingers
# ---------------------
def fingers_up(hand_landmarks):
    tips = [hand_landmarks.landmark[i] for i in FINGER_TIPS]
    fingers = []
    # Thumb
    fingers.append(1 if tips[0].x < hand_landmarks.landmark[2].x else 0)
    # Other fingers
    for i in range(1,5):
        fingers.append(1 if tips[i].y < hand_landmarks.landmark[i*4-2].y else 0)
    return fingers

# ---------------------
# Video Transformer
# ---------------------
class SignSpeechTransformer(VideoTransformerBase):
    def transform(self, frame):
        global last_word, last_time, sentence
        image = frame.to_ndarray(format="bgr24")
        frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        gesture_text = ""
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(image, handLms, mp_hands.HAND_CONNECTIONS)
                fingers = tuple(fingers_up(handLms))
                gesture_text = GESTURES.get(fingers, "")

                if gesture_text and (gesture_text != last_word or (time.time() - last_time > delay)):
                    if len(sentence) < MAX_WORDS:
                        sentence.append(gesture_text)
                    last_word = gesture_text
                    last_time = time.time()

        # Show info
        cv2.putText(image, "Word: " + (gesture_text if gesture_text else ""), (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.putText(image, "Sentence: " + " ".join(sentence), (10,70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        return image

# ---------------------
# Streamlit UI
# ---------------------
st.title("Sign-to-Speech Web App (Up to 4 Words)")
webrtc_streamer(key="sign-speech", video_transformer_factory=SignSpeechTransformer)

if st.button("Speak Sentence"):
    speak_text(" ".join(sentence))
    sentence = []
