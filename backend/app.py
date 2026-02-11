from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import gdown

app = Flask(__name__)
CORS(app)

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model will be saved directly inside backend folder
MODEL_PATH = os.path.join(BASE_DIR, "cnn_model.h5")

# 🔥 Google Drive File ID (PUT YOUR FILE ID HERE)
FILE_ID = "YOUR_FILE_ID_HERE"

# Download model if not present
if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    url = f"https://drive.google.com/file/d/1GVuVLtX3Bp4KTEmx1Y4EXaVlApktd2QB/view?usp=sharing"
    gdown.download(url, MODEL_PATH, quiet=False)

# Load model safely
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("Model loaded successfully!")

@app.route("/", methods=["GET"])
def home():
    return "CropGuard AI Backend Running"

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    # Preprocess image
    img = Image.open(file).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    preds = model.predict(img_array)
    confidence = float(np.max(preds[0])) * 100
    class_index = int(np.argmax(preds[0]))

    if confidence < 60:
        return jsonify({
            "disease": "Invalid or unclear image",
            "confidence": f"{confidence:.2f}%",
            "recommendation": "Please upload a clear crop leaf image."
        })

    return jsonify({
        "disease": f"Class {class_index}",
        "confidence": f"{confidence:.2f}%",
        "recommendation": "Apply appropriate treatment and monitor crop health."
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
