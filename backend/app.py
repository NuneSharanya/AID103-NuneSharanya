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


