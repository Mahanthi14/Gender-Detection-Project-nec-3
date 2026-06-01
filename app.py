import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
import json

# Load model
model = tf.keras.models.load_model("model/face_model.h5")

# Load labels
with open("model/labels.json", "r") as f:
    labels = json.load(f)

labels_inv = {v: k for k, v in labels.items()}

st.title("🎥 Live Face Recognition + Gender Detection")

run = st.checkbox("Start Webcam")

FRAME_WINDOW = st.image([])

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

while run:
    ret, frame = cap.read()
    if not ret:
        st.write("Camera error")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (128, 128))
        face_norm = face_resized / 255.0
        face_input = np.expand_dims(face_norm, axis=0)

        prediction = model.predict(face_input)
        idx = np.argmax(prediction)

        name = labels_inv[idx]

        # Simple gender logic
        if "Male" in name or "Rahul" in name or "Arjun" in name:
            gender = "Male"
        else:
            gender = "Female"

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.putText(frame, f"{name} | {gender}",
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2)

    FRAME_WINDOW.image(frame, channels="BGR")

cap.release()