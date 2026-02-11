from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import gdown

app = Flask(__name__)

# ✅ Enable CORS for frontend
CORS(app)

# ===============================
# Model Setup
# ===============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "cnn_model.h5")

# Download model if not exists
if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    FILE_ID = "1GVuVLtX3Bp4KTEmx1Y4EXaVlApktd2QB"
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("Model loaded successfully!")

# ===============================
# Routes
# ===============================

@app.route("/")
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

# ===============================
# Run
# ===============================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))