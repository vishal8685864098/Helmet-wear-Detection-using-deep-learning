import streamlit as st
import cv2
import os
import face_recognition
import torch
import numpy as np
from PIL import Image
from datetime import datetime
from twilio.rest import Client
import easyocr
from playsound import playsound
import pandas as pd

# ------------------- Paths -------------------
OFFENDERS_DB = "C:/Users/barig/yolov5/helmet detection/offenders_db"
SAVE_DIR = "C:/Users/barig/yolov5/helmet detection/detect/nonhelmet_offenders_demo"
WEIGHTS_PATH = "C:/Users/barig/yolov5/helmet detection/train/helmet_multi_class/weights/best.pt"
ALERT_SOUND_PATH = r"C:\Users\barig\yolov5\helmet detection\alert.mp3"

# ------------------- Twilio Setup -------------------
TWILIO_ACCOUNT_SID = "ACcd0cf5bd417ac5b199316d1d9ffad3f4"
TWILIO_AUTH_TOKEN = "7b26cebdb63b5f6509ff948e8fa4dadc"
TWILIO_PHONE_NUMBER = "+17655408179"
RECIPIENT_PHONE_NUMBER = "+918688586409"
client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# ------------------- EasyOCR -------------------
reader = easyocr.Reader(['en'])

# ------------------- Fine Log -------------------
fine_log = {}

# ------------------- Load YOLOv5 -------------------
@st.cache_resource
def load_model(weights_path):
    return torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path, force_reload=True)

# ------------------- Load Offenders -------------------
@st.cache_resource
def load_known_faces():
    known_encodings = []
    known_names = []
    for file in os.listdir(OFFENDERS_DB):
        if file.endswith(('.jpg', '.png')):
            image = face_recognition.load_image_file(os.path.join(OFFENDERS_DB, file))
            encoding = face_recognition.face_encodings(image)
            if encoding:
                known_encodings.append(encoding[0])
                known_names.append(os.path.splitext(file)[0])
    return known_encodings, known_names

# ------------------- Send Alert -------------------
def send_sms_alert(name):
    if name not in fine_log or not fine_log[name]['alert_sent']:
        message = client.messages.create(
            body=f"Alert! {name} detected without helmet again. Fine increased to ₹{fine_log[name]['fine']}",
            from_=TWILIO_PHONE_NUMBER,
            to=RECIPIENT_PHONE_NUMBER
        )
        fine_log[name]['alert_sent'] = True

# ------------------- Recognize Offenders -------------------
def recognize_offenders(frame, known_encodings, known_names):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            matched_idx = matches.index(True)
            name = known_names[matched_idx]

            if name not in fine_log:
                fine_log[name] = {'fine': 500, 'alert_sent': False}
            else:
                fine_log[name]['fine'] += 500

            send_sms_alert(name)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{name}_{timestamp}.jpg"
            save_path = os.path.join(SAVE_DIR, filename)
            face_img = frame[top:bottom, left:right].copy()
            cv2.imwrite(save_path, face_img)

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, f"{name} Fine:₹{fine_log[name]['fine']}", (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return frame

# ------------------- ANPR -------------------
def perform_anpr(frame, bbox):
    x1, y1, x2, y2 = bbox
    roi = frame[y1:y2, x1:x2]
    result = reader.readtext(roi)
    plate_text = " ".join([res[1] for res in result])
    return plate_text.strip()

# ------------------- Object Detection -------------------
def detect_objects(model, image):
    results = model(image)
    labels = results.xyxyn[0][:, -1].cpu().numpy()
    cords = results.xyxyn[0][:, :-1].cpu().numpy()
    return labels, cords

# ------------------- Streamlit Main -------------------
def main():
    st.title("Helmet Wear Detection + Face Recognition + ANPR + Alerts")
    source_type = st.sidebar.selectbox("Select Input", ["Webcam", "Image", "Video"])
    model = load_model(WEIGHTS_PATH)
    known_encodings, known_names = load_known_faces()
    detection_log = []

    if source_type == "Image":
        uploaded_image = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
        if uploaded_image:
            img = Image.open(uploaded_image)
            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            labels, cords = detect_objects(model, img_cv)

            for i, label in enumerate(labels):
                conf = cords[i][4]
                if conf > 0.5:
                    x1, y1, x2, y2 = (cords[i][:4] * np.array([img_cv.shape[1], img_cv.shape[0], img_cv.shape[1], img_cv.shape[0]])).astype(int)
                    class_id = int(label)
                    name = model.names[class_id]

                    if "helmet" in name.lower():
                        color = (0, 255, 0)
                        helmet_status = "Helmet"
                    else:
                        color = (0, 0, 255)
                        helmet_status = "No Helmet"
                        playsound(ALERT_SOUND_PATH)

                    cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(img_cv, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                    plate = perform_anpr(img_cv, (x1, y1, x2, y2)) if helmet_status == "No Helmet" else ""
                    detection_log.append({
                        "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Class": name,
                        "Helmet Status": helmet_status,
                        "ANPR Number": plate
                    })

            img_result = recognize_offenders(img_cv, known_encodings, known_names)
            st.image(cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB), channels="RGB")

    elif source_type == "Webcam":
        cap = cv2.VideoCapture(0)
        stframe = st.empty()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            labels, cords = detect_objects(model, frame)
            for i, label in enumerate(labels):
                conf = cords[i][4]
                if conf > 0.5:
                    x1, y1, x2, y2 = (cords[i][:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])).astype(int)
                    class_id = int(label)
                    name = model.names[class_id]

                    if "helmet" in name.lower():
                        color = (0, 255, 0)
                        helmet_status = "Helmet"
                    else:
                        color = (0, 0, 255)
                        helmet_status = "No Helmet"
                        playsound(ALERT_SOUND_PATH)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                    plate = perform_anpr(frame, (x1, y1, x2, y2)) if helmet_status == "No Helmet" else ""
                    detection_log.append({
                        "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Class": name,
                        "Helmet Status": helmet_status,
                        "ANPR Number": plate
                    })

            frame = recognize_offenders(frame, known_encodings, known_names)
            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
        cap.release()

    elif source_type == "Video":
        uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
        if uploaded_video:
            tfile = "temp_video.mp4"
            with open(tfile, 'wb') as f:
                f.write(uploaded_video.read())

            cap = cv2.VideoCapture(tfile)
            stframe = st.empty()
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                labels, cords = detect_objects(model, frame)
                for i, label in enumerate(labels):
                    conf = cords[i][4]
                    if conf > 0.5:
                        x1, y1, x2, y2 = (cords[i][:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])).astype(int)
                        class_id = int(label)
                        name = model.names[class_id]

                        if "helmet" in name.lower():
                            color = (0, 255, 0)
                            helmet_status = "Helmet"
                        else:
                            color = (0, 0, 255)
                            helmet_status = "No Helmet"
                            playsound(ALERT_SOUND_PATH)

                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                        plate = perform_anpr(frame, (x1, y1, x2, y2)) if helmet_status == "No Helmet" else ""
                        detection_log.append({
                            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "Class": name,
                            "Helmet Status": helmet_status,
                            "ANPR Number": plate
                        })

                frame = recognize_offenders(frame, known_encodings, known_names)
                stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
            cap.release()

    # ------------------- Log Export -------------------
    if detection_log:
        df = pd.DataFrame(detection_log)
        st.subheader("Detection Logs")
        st.dataframe(df)

        csv = df.to_csv(index=False).encode()
        st.download_button(
            label="Download Logs as CSV",
            data=csv,
            file_name='helmet_detection_logs.csv',
            mime='text/csv',
        )

if __name__ == "__main__":
    main()
