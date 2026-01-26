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
    """
    Convert image bytes to preprocessed numpy array
    """
    try:
        # Convert bytes to PIL Image
        img = Image.open(io.BytesIO(image_bytes))
        img = img.convert('RGB')
        
        # Convert to numpy array and resize
        img_array = np.array(img)
        img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        
        # Normalize
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

# =============================
# PREDICTION FUNCTION
# =============================
def predict_disease(image_bytes, crop_name):
    """
    Predict disease from image for specified crop
    Returns: {disease, confidence, all_predictions}
    """
    if not model:
        return {"error": "Model not loaded", "status": 500}
    
    img_array = preprocess_image(image_bytes)
    if img_array is None:
        return {"error": "Invalid image format", "status": 400}
    
    try:
        # Get predictions
        preds = model.predict(img_array)[0]
        
        # Filter by crop (if crop specified)
        if crop_name in crop_map:
            filtered_preds = {
                class_names[i]: float(preds[i])
                for i in range(len(class_names))
                if class_names[i] in crop_map[crop_name]
            }
        else:
            # If crop not found, use all predictions
            filtered_preds = {
                class_names[i]: float(preds[i])
                for i in range(len(class_names))
            }
        
        # Get top prediction
        if filtered_preds:
            disease = max(filtered_preds, key=filtered_preds.get)
            confidence = filtered_preds[disease] * 100
        else:
            disease = "Unknown"
            confidence = 0.0
        
        # Extract just the disease part (remove crop name)
        if "___" in disease:
            disease_only = disease.split("___", 1)[1]
        else:
            disease_only = disease
        
        return {
            "status": 200,
            "disease": disease_only,
            "confidence": round(confidence, 2),
            "full_classification": disease,
            "all_predictions": filtered_preds
        }
    
    except Exception as e:
        print(f"Prediction error: {e}")
        return {"error": str(e), "status": 500}

# =============================
# ROUTES
# =============================

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    POST endpoint for disease prediction
    
    Expected input:
    - multipart/form-data with:
      - image: image file
      - crop: crop name (string)
    
    Returns:
    {
        "disease": "disease name",
        "confidence": 85.5,
        "full_classification": "Crop___Disease",
        "crop": "Tomato"
    }
    """
    try:
        # Check if image file is present
        if 'image' not in request.files:
            return jsonify({"error": "Image file required"}), 400
        
        image_file = request.files['image']
        crop_name = request.form.get('crop', 'Unknown')
        
        if image_file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        
        # Read image bytes
        image_bytes = image_file.read()
        
        # Make prediction
        result = predict_disease(image_bytes, crop_name)
        
        if "error" in result:
            return jsonify(result), result.get("status", 500)
        
        return jsonify({
            "success": True,
            "crop": crop_name,
            "disease": result["disease"],
            "confidence": result["confidence"],
            "full_classification": result["full_classification"]
        }), 200
    
    except Exception as e:
        print(f"API error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """
    GET available crops and diseases
    """
    try:
        crops = {}
        for crop in crop_map:
            diseases = [d.split("___")[1] for d in crop_map[crop]]
            crops[crop] = list(set(diseases))
        
        return jsonify({
            "success": True,
            "crops": crops
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/crops', methods=['GET'])
def get_crops():
    """
    GET available crops and diseases
    """
    try:
        crops = {}
        for crop in crop_map:
            diseases = [d.split("___")[1] for d in crop_map[crop] if "___" in d]
            crops[crop] = list(set(diseases))
        
        return jsonify({
            "success": True,
            "crops": crops,
            "total_crops": len(crops),
            "total_classes": len(class_names)
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# =============================
# ERROR HANDLERS
# =============================

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error"}), 500

# =============================
# MAIN
# =============================

if __name__ == '__main__':
    print("🚀 Starting Flask API...")
    print(f"📦 Model path: {MODEL_PATH}")
    print(f"🌾 Total classes: {len(class_names)}")
    print(f"🥕 Total crops: {len(crop_map)}")
    app.run(host='0.0.0.0', port=5001, debug=False, use_reloader=False)

