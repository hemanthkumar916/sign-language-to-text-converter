from flask import Flask, render_template, Response, jsonify, request, redirect, url_for
import cv2
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
from PIL import Image

app = Flask(__name__)

# Load model and label map
ANNOTATION_PATH = "C:/Users/khema/RealTimeObjectDetection-main/Tensorflow/workspace/annotations"
MODEL_DIR = "C:/Users/khema/RealTimeObjectDetection-main/Tensorflow/workspace/exported-models/my_ssd_mobnet/saved_model"

# Load TensorFlow model
detect_fn = tf.saved_model.load(MODEL_DIR)

# Load label map
category_index = label_map_util.create_category_index_from_labelmap(ANNOTATION_PATH + '/label_map.pbtxt')

# Global variables
latest_detected_sign = None
stop_video = False
pause_video = True
selected_language = 'E'  # Default language: English

# Translations for detected gestures
translations = {
    "hello": {"E": "Hello", "T": "హలో", "H": "नमस्ते"},
    "yes": {"E": "Yes", "T": "అవును", "H": "हाँ"},
    "no": {"E": "No", "T": "కాదు", "H": "नहीं"},
    "thanks": {"E": "Thanks", "T": "ధన్యవాదాలు", "H": "धन्यवाद"},
    "i love you": {"E": "I love you", "T": "నేను నిన్ను ప్రేమిస్తున్నాను", "H": "मैं तुमसे प्यार करता हूँ"},
    "help": {"E": "Help", "T": "సహాయం", "H": "मदद"},
    "together": {"E": "Together", "T": "కలిసి", "H": "साथ"},
    "good luck": {"E": "Good luck", "T": "శుభవార్త", "H": "शुभकामनाएँ"},
    "good": {"E": "Good", "T": "మంచి", "H": "अच्छा"},
    "please": {"E": "Please", "T": "దయచేసి", "H": "कृपया"},
    "hope": {"E": "Hope", "T": "ఆశ", "H": "आशा"},
    "danger": {"E": "Danger", "T": "ప్రమాదం", "H": "खतरा"}
}

def detect_objects(image):
    """
    Detect objects in the given image using the loaded TensorFlow model.
    """
    image = tf.convert_to_tensor(image, dtype=tf.uint8)
    input_tensor = tf.expand_dims(image, axis=0)
    detections = detect_fn(input_tensor)
    return detections

def generate_frames():
    """
    Generate video frames with real-time sign detection.
    """
    global latest_detected_sign, stop_video, pause_video, selected_language
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    while not stop_video:
        if not pause_video:
            ret, frame = cap.read()
            if not ret:
                break

            # Perform object detection
            detections = detect_objects(frame)

            # Find the label with the highest confidence score
            detected_label = None
            highest_score = 0.0

            for i in range(int(detections['num_detections'][0])):
                score = detections['detection_scores'][0][i].numpy()
                if score > highest_score and score > 0.5:  # Only consider detections with a confidence score > 50%
                    highest_score = score
                    class_id = int(detections['detection_classes'][0][i].numpy())
                    detected_label = category_index[class_id]['name']

            # Update the latest detected sign
            latest_detected_sign = detected_label

            # Get translation based on selected language
            translated_text = translations.get(detected_label, {}).get(selected_language, detected_label)

            # Draw the detected label on the frame
            if detected_label:
                cv2.putText(frame, f"Detected: {translated_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Convert the frame to JPEG format
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Yield the frame in byte format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            # If paused, yield a black frame
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + cv2.imencode('.jpg', np.zeros((480, 640, 3), dtype=np.uint8))[1].tobytes() + b'\r\n')

    # Release the webcam
    cap.release()
    stop_video = False  # Reset the stop flag

@app.route('/')
def intro():
    """
    Render the introductory page.
    """
    return render_template('intro.html')

@app.route('/home')
def home():
    """
    Render the homepage with the HTML template.
    """
    return render_template('index.html')

@app.route('/video_detection')
def video_detection():
    """
    Render the real-time video detection page.
    """
    return render_template('video_detection.html')

@app.route('/video_feed')
def video_feed():
    """
    Route to stream the video feed.
    """
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_detected_sign')
def get_detected_sign():
    """
    Route to send the latest detected sign to the client.
    """
    global latest_detected_sign, selected_language
    translated_text = translations.get(latest_detected_sign, {}).get(selected_language, latest_detected_sign)
    return jsonify({'sign': translated_text})

@app.route('/set_language', methods=['POST'])
def set_language():
    """
    Route to set the selected language.
    """
    global selected_language
    selected_language = request.json['language']
    return jsonify({'status': 'success', 'language': selected_language})

@app.route('/start_detection')
def start_detection():
    """
    Route to start the video detection.
    """
    global pause_video
    pause_video = False
    return '', 204

@app.route('/pause_detection')
def pause_detection():
    """
    Route to pause the video detection.
    """
    global pause_video
    pause_video = True
    return '', 204

@app.route('/stop_video')
def stop_video():
    """
    Route to stop the video detection.
    """
    global stop_video
    stop_video = True
    return redirect(url_for('home'))

@app.route('/image_detection')
def image_detection():
    """
    Render the image detection page.
    """
    return render_template('image_detection.html')

@app.route('/detect_image', methods=['POST'])
def detect_image():
    """
    Handle image upload and perform object detection.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty file uploaded'}), 400

    try:
        # Convert uploaded file to NumPy array
        image = np.array(Image.open(file))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    except Exception as e:
        return jsonify({'error': f'Failed to process image: {str(e)}'}), 400

    # Perform object detection
    detections = detect_objects(image)

    # Find the label with the highest confidence score
    detected_label = None
    highest_score = 0.0

    for i in range(int(detections['num_detections'][0])):
        score = detections['detection_scores'][0][i].numpy()
        if score > highest_score and score > 0.5:  # Only consider detections with a confidence score > 50%
            highest_score = score
            class_id = int(detections['detection_classes'][0][i].numpy())
            detected_label = category_index[class_id]['name']

    if detected_label:
        return jsonify({'label': detected_label})
    else:
        return jsonify({'error': 'No objects detected'}), 400

if __name__ == '__main__':
    app.run(debug=True)