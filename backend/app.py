from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)
CORS(app)   # 🔴 REQUIRED for frontend access

# Load model
model = tf.keras.models.load_model("../models/cnn_model.h5")

@app.route("/", methods=["GET"])
def home():
    return "CropGuard AI Backend Running"

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    img = Image.open(file).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)
    class_index = int(np.argmax(preds[0]))
    confidence = float(np.max(preds[0]))

    return jsonify({
        "disease": f"Class {class_index}",
        "confidence": f"{confidence*100:.2f}%",
        "recommendation": "Apply appropriate treatment and monitor crop health."
    })

if __name__ == "__main__":
    app.run(debug=True)

'''from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import cv2

app = Flask(__name__)

# Load trained model
model = tf.keras.models.load_model("../models/cnn_model.h5")

# Class names (ORDER MUST MATCH DATASET)
CLASS_NAMES = [
    "Pepper Bell Bacterial Spot",
    "Pepper Bell Healthy",
    "Potato Early Blight",
    "Potato Healthy",
    "Potato Late Blight",
    "Tomato Bacterial Spot",
    "Tomato Early Blight",
    "Tomato Late Blight",
    "Tomato Leaf Mold",
    "Tomato Septoria Leaf Spot",
    "Tomato Spider Mites",
    "Tomato Target Spot",
    "Tomato Mosaic Virus",
    "Tomato Yellow Leaf Curl Virus",
    "Tomato Healthy"
]

# Treatment mapping
TREATMENT_MAP = {
    "Tomato Late Blight": "Apply fungicide immediately and remove infected plants.",
    "Tomato Early Blight": "Use copper-based fungicide and avoid overhead irrigation.",
    "Tomato Mosaic Virus": "Remove infected plants and control insect vectors.",
    "Tomato Yellow Leaf Curl Virus": "Control whiteflies and remove infected crops.",
    "Potato Late Blight": "Apply protective fungicides and improve field drainage.",
    "Potato Early Blight": "Use disease-resistant varieties and apply fungicide if needed.",
    "Pepper Bell Bacterial Spot": "Use copper sprays and avoid working in wet fields."
}

# Image preprocessing
def preprocess_image(image_path, img_size=224):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (img_size, img_size))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Severity logic
def get_severity(confidence):
    if confidence >= 80:
        return "High"
    elif confidence >= 50:
        return "Medium"
    else:
        return "Low"

# Explainable AI text
def generate_explanation(disease, confidence, severity):
    return (
        f"The AI analyzed the crop image and detected {disease} "
        f"with a confidence of {confidence:.2f}%. "
        f"The disease severity is assessed as {severity}. "
        f"Recommended treatment and preventive measures are provided accordingly."
    )
@app.route("/", methods=["GET"])
def home():
    return "CropGuard AI Backend Running"

# Prediction API
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files["file"]
    img = Image.open(file).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    class_index = int(np.argmax(predictions[0]))
    confidence = float(np.max(predictions[0]))

    return jsonify({
        "disease": str(class_index),
        "confidence": f"{confidence*100:.2f}%",
        "recommendation": "Apply recommended treatment and monitor crop health."
    })

'''
@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    image_path = 'temp.jpg'
    file.save(image_path)

    image = preprocess_image(image_path)
    prediction = model.predict(image)

    confidence = float(np.max(prediction)) * 100
    disease_index = np.argmax(prediction)
    disease_name = CLASS_NAMES[disease_index]

    severity = get_severity(confidence)
    treatment = TREATMENT_MAP.get(disease_name, "General crop care is recommended.")
    explanation = generate_explanation(disease_name, confidence, severity)

    return jsonify({
        "Disease": disease_name,
        "Confidence (%)": round(confidence, 2),
        "Severity": severity,
        "Treatment": treatment,
        "AI_Explanation": explanation
    })
'''
if __name__ == '__main__':
    app.run(debug=True)
'''