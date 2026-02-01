"""
Flask API for crop disease detection model
Loads .h5 model and exposes prediction endpoint
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import cv2
import os
from collections import defaultdict
import io
from PIL import Image

# =============================
# APP SETUP
# =============================
app = Flask(__name__)
CORS(app)

# =============================
# CONFIGURATION
# =============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "crop_disease_model.h5")
IMG_SIZE = 224

# =============================
# LOAD MODEL
# =============================
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"✅ Model loaded from {MODEL_PATH}")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    model = None

# =============================
# CLASS NAMES
# =============================
class_names = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___healthy",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___healthy",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___healthy",
    "Potato___Late_blight",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___healthy",
    "Strawberry___Leaf_scorch",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_healthy",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_mosaic_virus",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___healthy",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus"
]

# =============================
# BUILD CROP → DISEASE MAP
# =============================
crop_map = defaultdict(list)
for cls in class_names:
    crop = cls.split("___")[0]
    crop_map[crop].append(cls)

# =============================
# IMAGE PREPROCESSING
# =============================
def preprocess_image(image_bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_array = np.array(img)
        img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        print(f"❌ Image preprocessing error: {e}")
        return None

# =============================
# PREDICTION LOGIC
# =============================
def predict_disease(image_bytes, crop_name):
    if model is None:
        return {"error": "Model not loaded", "status": 500}

    img_array = preprocess_image(image_bytes)
    if img_array is None:
        return {"error": "Invalid image", "status": 400}

    preds = model.predict(img_array)[0]

    if crop_name in crop_map:
        filtered = {
            class_names[i]: float(preds[i])
            for i in range(len(class_names))
            if class_names[i] in crop_map[crop_name]
        }
    else:
        filtered = {class_names[i]: float(preds[i]) for i in range(len(class_names))}

    disease = max(filtered, key=filtered.get)
    confidence = filtered[disease] * 100
    disease_only = disease.split("___", 1)[1] if "___" in disease else disease

    return {
        "status": 200,
        "disease": disease_only,
        "confidence": round(confidence, 2),
        "full_classification": disease
    }

# =============================
# ROUTES
# =============================

@app.route("/")
def root():
    return {
        "status": "Agro ML API running",
        "model_loaded": model is not None,
        "total_classes": len(class_names),
        "total_crops": len(crop_map)
    }

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None
    })

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Image file required"}), 400

    image_file = request.files["image"]
    crop_name = request.form.get("crop", "Unknown")

    if image_file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    image_bytes = image_file.read()
    result = predict_disease(image_bytes, crop_name)

    if "error" in result:
        return jsonify(result), result.get("status", 500)

    return jsonify({
        "success": True,
        "crop": crop_name,
        "disease": result["disease"],
        "confidence": result["confidence"],
        "full_classification": result["full_classification"]
    })

@app.route("/crops", methods=["GET"])
def get_crops():
    crops = {}
    for crop in crop_map:
        diseases = [d.split("___")[1] for d in crop_map[crop] if "___" in d]
        crops[crop] = list(set(diseases))

    return jsonify({
        "success": True,
        "crops": crops,
        "total_crops": len(crops),
        "total_classes": len(class_names)
    })

# =============================
# ERROR HANDLERS
# =============================
@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error"}), 500
