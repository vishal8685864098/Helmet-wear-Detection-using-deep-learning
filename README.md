# 🛵 🛵🛵Helmet Wear Detection Using Deep Learning

This project uses YOLOv5 and OpenCV to detect whether a person riding a two-wheeler is wearing a helmet or not. It also integrates real-time **face recognition**, **automatic number plate recognition (ANPR)** using EasyOCR, **fine logging**, **helmet color detection**, and **Twilio SMS alerts** for repeated offenders.

---

## 📌 Features

- ✅ Helmet and non-helmet detection using YOLOv5
- 🟩 Draws green bounding box for helmet wearers
- 🟥 Draws red bounding box for non-helmet wearers
- 🔁 Real-time webcam, image, and video input support
- 🧠 Face recognition for identifying repeat offenders
- 🔢 Number Plate Recognition using EasyOCR (ANPR)
- 🧾 Fine tracking & logging (CSV export)
- 📦 Helmet color classification
- 🔔 Sound alert & Twilio SMS for violations
- 📊 Streamlit Dashboard UI for live monitoring

---

## 🧠 Technologies Used

| Technology       | Purpose                              |
|------------------|---------------------------------------|
| YOLOv5           | Helmet detection                      |
| OpenCV           | Image & video processing              |
| EasyOCR          | ANPR (Number plate reading)           |
| face_recognition | Identifying known offenders           |
| Twilio API       | SMS alerts                            |
| Streamlit        | UI and visualization dashboard        |
| Pandas           | Logging fines and data analysis       |

---

## 🖼️ Demo

_Add screenshots or screen recordings here (optional)_  
![Helmet Detection](images/demo1.png)
<img width="807" height="730" alt="Screenshot 2025-06-10 090618" src="https://github.com/user-attachments/assets/93d5d08d-0edc-4d90-9ef5-dd68b749e682" />

<img width="604" height="679" alt="Screenshot 2025-06-10 090836" src="https://github.com/user-attachments/assets/3c5de05a-870f-4901-9dc4-8a99f8b20107" />
<img width="736" height="574" alt="Screenshot 2025-06-10 090911" src="https://github.com/user-attachments/assets/8c303926-dda0-4237-a107-adfa72524cce" />
<img width="923" height="729" alt="Screenshot 2025-06-10 112034" src="https://github.com/user-attachments/assets/f18ecd77-0060-49a8-bd1d-801cf5aa931e" />
<img width="923" height="729" alt="Screenshot 2025-06-10 112034" src="https://github.com/user-attachments/assets/71711046-3599-4369-af90-44d1d4ef9269" />





---

## 📁 Folder Structure
helmet-detection-project/
├── detect/ # Output results
├── offenders_db/ # Known offenders (images)
├── train/ # YOLOv5 training data
├── yolov5/ # YOLOv5 model repo
├── app.py # Streamlit app
├── requirements.txt # Dependencies
└── README.md # Project documentation




---

## ⚙️ Installation & Setup

1. **Clone the repository**
```bash
git clone https://github.com/vishal8685864098/helmet-detection-project.git
cd helmet-detection-project

2.Create virtual environment & activate

bash
Copy code


bash
Copy code
python -m venv venv
venv\Scripts\activate   # For Windows

3.Install dependencies

bash
Copy code
pip install -r requirements.txt

4.Run the Streamlit app

bash
Copy code
streamlit run app.py

📥 Dataset
Due to size limitations, the dataset is not included in this repo.
👉 Download the dataset here and place it in the dataset/ folder.
dataset link->  https://universe.roboflow.com/gw-khadatkar-and-sv-wasule/helmet-and-no-helmet-rider-detection/dataset/6

🔐 Offender Face DB
Place known offender images in the folder:

Copy code
offenders_db/
├── offender1.jpg
├── offender2.jpg
Each face will be matched during detection to trigger repeated offense warnings and send SMS alerts.



🔔 Twilio SMS Setup (Optional if you want a real time alert notification)
1.Create a Twilio account

2.Get your Account SID, Auth Token, and Twilio phone number

3.Add these to a .env file:

env
Copy code
TWILIO_SID=your_sid
TWILIO_AUTH_TOKEN=your_token
TWILIO_PHONE=+1234567890
MY_PHONE=+91xxxxxxxxxx

📊 Logs and Fines
All detections are logged in a CSV file with:

Timestamp

Offender name

Number plate

Helmet status

Fine amount

📈 Helmet Color Detection
Detects the helmet color using basic color filtering on the detected helmet region. Can be used for analytics or pattern tracking.

🧾 Fine Rules
Condition	Fine
First offense	₹500
Repeated offense	₹1000+
No helmet + no plate	₹1500

⭐️ Show Your Support
If you find this project helpful, please ⭐️ the repo and share it with others!

📄 License
This project is licensed under the MIT License.

yaml
Copy code

---

### 👉 Next Steps

1. **Copy** the above into a file named `README.md` inside your project folder.
2. **Add and commit** it:
```bash
git add README.md
git commit -m "Add project README"
git push



