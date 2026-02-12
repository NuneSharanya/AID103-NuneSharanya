from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import gdown

app = Flask(__name__)
CORS(app)  # allow all origins for now

# ===============================
# Model Setup
# ===============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "cnn_model.h5")

# Download model if missing
if not os.path.exists(MODEL_PATH):
    print("⬇️ Downloading model from Google Drive...")
    FILE_ID = "1GVuVLtX3Bp4KTEmx1Y4EXaVlApktd2QB"
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)

print("📦 Loading model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("✅ Model loaded successfully")

# ===============================
# Routes
# ===============================

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "CropGuard AI Backend Running"})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]

        if file.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        # Preprocess image
        img = Image.open(file).convert("RGB")
        img = img.resize((224, 224))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        preds = model.predict(img_array)
        confidence = float(np.max(preds[0])) * 100
        class_index = int(np.argmax(preds[0]))

        if confidence < 60:
            return jsonify({
                "disease": "Unclear image",
                "confidence": f"{confidence:.2f}%",
                "recommendation": "Upload a clear crop leaf image"
            })

        return jsonify({
            "disease": f"Class {class_index}",
            "confidence": f"{confidence:.2f}%",
            "recommendation": "Apply appropriate treatment and mointor crop health regularly"
        })

    except Exception as e:
        print("❌ Prediction error:", e)
        return jsonify({
            "error": "Prediction failed",
            "details": str(e)
        }), 500

# ===============================
# Run
# ===============================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)