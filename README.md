# 🤟 Sign Language to Text Converter

This project converts American Sign Language (ASL) gestures into readable text using deep learning and computer vision techniques. It is built using TensorFlow, OpenCV, and Flask and supports both real-time webcam detection and image uploads.

---

## 🧠 Project Highlights

- 🎯 **Real-time Sign Detection** via webcam
- 🖼️ **Static Image Gesture Recognition**
- 🌐 **Multilingual Support**: English, Hindi, Telugu
- ⚡ **Built with TensorFlow's Object Detection API (SSD MobileNet V2)**
- 🧩 **User-friendly Flask Web Interface**
- 📦 **Offline Capable** – No internet needed once set up

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

Visit `http://localhost:5000` in your browser.

---

## 🖼️ Supported Gestures

- `hello`, `yes`, `no`, `thanks`, `i love you`, `help`, `together`, `good luck`, `good`, `please`, `hope`, `danger`

---

## 📁 Project Structure

```
├── app.py                    # Flask web app
├── train.py                 # Model training script
├── Tensorflow/              # Contains model config and checkpoints
├── templates/               # HTML files for web interface
├── static/                  # CSS, JS, etc.
├── requirements.txt         # Python dependencies
└── README.md
```

---

## 🧑‍💻 Author

**Keta Hemanth Kumar**  
MCA Graduate, Vignan’s Institute of Information Technology

---

## 📜 License

This project is for educational and academic use.