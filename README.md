# 🤟 Sign Language to Text Converter

This project converts American Sign Language (ASL) gestures into readable text using deep learning and computer vision techniques. It is built using TensorFlow, OpenCV, and Flask, and supports both real-time webcam detection and static image uploads.

---

## 🧠 Project Highlights

- 🎯 Real-time sign detection via webcam
- 🖼️ Static image gesture recognition
- 🌐 Multilingual support: English, Hindi, Telugu
- ⚡ Built with TensorFlow's Object Detection API (SSD MobileNet V2)
- 🧩 User-friendly Flask web interface
- 📦 Offline Capable – No internet needed after setup

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the application

```bash
python app.py
```

Open your browser and go to `http://localhost:5000`

---

## 🖼️ Application Screenshots

### 🔹 Web Interface (Home Page)
![Home](assets/Screenshot_2025-04-29_190014.png)

### 🔹 Image Detection - Before & After
![Image Input](assets/Screenshot_2025-04-29_190032.png)
![Image Output](assets/Screenshot_2025-04-29_190100.png)

### 🔹 Real-Time Video Detection
![Video Detection](assets/Screenshot_2025-04-29_191719.png)
![Video Detection Idle](assets/Screenshot_2025-04-29_185927.png)

### 🔹 Intro Page
![Intro Page](assets/Screenshot_2025-04-29_185910.png)

---

## 📁 Project Structure

```
├── app.py                    # Flask app
├── train.py                 # Training script
├── Tensorflow/              # Model files and configs
├── templates/               # HTML templates
├── static/                  # CSS/JS files
├── assets/                  # Demo images for GitHub display
├── requirements.txt         # Python dependencies
├── .gitignore
├── LICENSE
└── README.md
```

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🧑‍💻 Author

**Keta Hemanth Kumar**  
MCA (2023–2025), Vignan’s Institute of Information Technology  
Final Year Project: Sign Language to Text Converter