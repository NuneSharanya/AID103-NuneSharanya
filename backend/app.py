from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)
CORS(app)

# Load model safely using absolute path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "cnn_model.h5")

# ✅ SAFE LOAD
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

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

    # 🔒 Confidence threshold check
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

