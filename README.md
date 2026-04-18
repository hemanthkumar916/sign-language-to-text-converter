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
<img src="https://github.com/hemanthkumar916/sign-language-to-text-converter/raw/main/assets/Screenshot 2025-04-29 185927.png" alt="Home" width="600" />

### 🔹 Image Detection - Before & After  
<img src="https://github.com/hemanthkumar916/sign-language-to-text-converter/raw/main/assets/Screenshot 2025-04-29 190032.png" alt="Image Input" width="600" />  
<img src="https://github.com/hemanthkumar916/sign-language-to-text-converter/raw/main/assets/Screenshot 2025-04-29 190100.png" alt="Image Output" width="600" />

### 🔹 Real-Time Video Detection 
<img src="https://github.com/hemanthkumar916/sign-language-to-text-converter/raw/main/assets/Screenshot 2025-04-29 190014.png" alt="Video Detection Idle" width="600" />

### 🔹 Intro Page  
<img src="https://github.com/hemanthkumar916/sign-language-to-text-converter/raw/main/assets/Screenshot 2025-04-29 185910.png" alt="Intro Page" width="600" />

---

## 📁 Project Structure

```
├── app.py                    # Flask app
├── train.py                  # Training script
├── Tensorflow/               # Model files and configs
├── templates/                # HTML templates
├── static/                   # CSS/JS files
├── assets/                   # Demo images for GitHub display
├── requirements.txt          # Python dependencies
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
