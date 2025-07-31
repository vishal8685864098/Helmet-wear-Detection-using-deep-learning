import streamlit as st
import cv2
import numpy as np
import os
import datetime
import csv
import easyocr
from PIL import Image
from twilio.rest import Client
import tempfile

# Commented out for Streamlit Cloud (dlib not supported)
# import face_recognition

# Set page config
st.set_page_config(page_title="Helmet Detection App", layout="wide")

# Load YOLOv5 model
import torch
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)

# Define labels
labels = ['helmet', 'no-helmet']

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Commented out for Streamlit Cloud (dlib not supported)
"""
def load_known_faces(known_faces_dir='offenders_db'):
    known_face_encodings = []
    known_face_names = []
    for filename in os.listdir(known_faces_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(known_faces_dir, filename)
            image = face_recognition.load_image_file(image_path)
            encoding = face_recognition.face_encodings(image)
            if encoding:
                known_face_encodings.append(encoding[0])
                known_face_names.append(os.path.splitext(filename)[0])
    return known_face_encodings, known_face_names
"""

# Function to log violations
def log_violation(helmet_status, plate_number="Unknown", offender_name="Unknown"):
    file_exists = os.path.isfile("violation_log.csv")
    with open("violation_log.csv", mode="a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["DateTime", "Helmet_Status", "Plate_Number", "Offender_Name", "Fine"])
        fine = 500 if helmet_status == "no-helmet" else 0
        writer.writerow([datetime.datetime.now(), helmet_status, plate_number, offender_name, fine])

# Function to extract number plate
def extract_plate_number(cropped_region):
    result = reader.readtext(cropped_region)
    text = ""
    for detection in result:
        bbox, detected_text, confidence = detection
        if confidence > 0.4:
            text += detected_text + " "
    return text.strip()

# Commented out for Streamlit Cloud (dlib not supported)
"""
def recognize_offenders(frame, known_face_encodings, known_face_names):
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    
    names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
        names.append(name)
    return names
"""

# SMS Notification (Optional)
def send_sms(to_number, message):
    try:
        account_sid = "your_twilio_sid"
        auth_token = "your_twilio_auth_token"
        client = Client(account_sid, auth_token)
        message = client.messages.create(
            body=message,
            from_="+Your_Twilio_Number",
            to=to_number
        )
    except Exception as e:
        st.warning(f"SMS not sent: {e}")

# Streamlit app UI
st.title("🪖 Helmet Detection App with YOLOv5")
option = st.sidebar.selectbox("Select Input Type", ("Webcam", "Image", "Video"))

if option == "Image":
    image_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if image_file:
        img = Image.open(image_file)
        img_np = np.array(img)
        results = model(img_np)
        boxes = results.pandas().xyxy[0]
        for index, row in boxes.iterrows():
            label = row['name']
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            color = (0, 255, 0) if label == "helmet" else (0, 0, 255)
            cv2.rectangle(img_np, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img_np, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        st.image(img_np, channels="BGR", caption="Detected Image")

elif option == "Video":
    video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if video_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        cap = cv2.VideoCapture(tfile.name)

        stframe = st.empty()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame)
            boxes = results.pandas().xyxy[0]
            for index, row in boxes.iterrows():
                label = row['name']
                x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
                color = (0, 255, 0) if label == "helmet" else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            stframe.image(frame, channels="BGR")

        cap.release()

elif option == "Webcam":
    run = st.checkbox("Start Webcam")
    FRAME_WINDOW = st.image([])
    cap = cv2.VideoCapture(0)

    # Commented for Streamlit Cloud
    # known_encodings, known_names = load_known_faces()

    while run:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        boxes = results.pandas().xyxy[0]
        for index, row in boxes.iterrows():
            label = row['name']
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            color = (0, 255, 0) if label == "helmet" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # Optional: Save and notify if no helmet
            if label == "no-helmet":
                crop = frame[y1:y2, x1:x2]
                plate_number = extract_plate_number(crop)
                offender_name = "Unknown"  # or from face recognition
                log_violation(label, plate_number, offender_name)
                # send_sms("+91xxxxxxxxxx", f"Helmet Violation Detected! Plate: {plate_number}")

        FRAME_WINDOW.image(frame, channels="BGR")
    cap.release()
