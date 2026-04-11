"""
ml_service.py
-------------
Central ML service for AgriPredict.
Handles all three core ML features:
  1. Crop Yield Prediction     → predict_yield()
  2. Crop Recommendation       → recommend_crop()
  3. Plant Disease Detection   → detect_disease()

All models are loaded once at startup and reused across requests.
No external API keys required — everything runs locally.
"""

import os
import io
import joblib
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ─────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR   = os.path.join(BASE_DIR, "ml_models")

# Yield model
YIELD_MODEL_PATH         = os.path.join(MODELS_DIR, "yield_model.pkl")
ENCODER_CROP_PATH        = os.path.join(MODELS_DIR, "encoder_crop.pkl")
ENCODER_SEASON_PATH      = os.path.join(MODELS_DIR, "encoder_season.pkl")
ENCODER_STATE_PATH       = os.path.join(MODELS_DIR, "encoder_state.pkl")

# Crop recommendation model
CROP_REC_MODEL_PATH      = os.path.join(MODELS_DIR, "crop_rec_model.pkl")
CROP_REC_SCALER_PATH     = os.path.join(MODELS_DIR, "crop_rec_scaler.pkl")
CROP_REC_ENCODER_PATH    = os.path.join(MODELS_DIR, "crop_rec_label_encoder.pkl")

# Disease detection model
DISEASE_MODEL_PATH       = os.path.join(MODELS_DIR, "plant_disease_model.pth")


# ─────────────────────────────────────────────────────────────────
# Lazy-loaded model holders (loaded once, reused forever)
# ─────────────────────────────────────────────────────────────────
_yield_model         = None
_encoder_crop        = None
_encoder_season      = None
_encoder_state       = None

_crop_rec_model      = None
_crop_rec_scaler     = None
_crop_rec_encoder    = None

_disease_model       = None


# ─────────────────────────────────────────────────────────────────
# Disease detection constants
# ─────────────────────────────────────────────────────────────────
CLASS_NAMES = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy",
    "Grape___Black_rot", "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy",
    "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch", "Strawberry___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
    "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

DISEASE_RECOMMENDATIONS = {
    "Apple___Apple_scab": {
        "cause": "Fungal infection caused by Venturia inaequalis, thrives in wet/cool conditions.",
        "pesticide": "Captan 50 WP or Mancozeb 75 WP",
        "dosage": "2.5g per litre of water",
        "prevention": "Prune infected branches, ensure good air circulation, avoid overhead irrigation.",
        "treatment_timeline": "Spray every 7-10 days during wet season"
    },
    "Apple___Black_rot": {
        "cause": "Fungal disease caused by Botryosphaeria obtusa, spreads via infected wood.",
        "pesticide": "Thiophanate-methyl or Captan",
        "dosage": "2g per litre of water",
        "prevention": "Remove mummified fruits, prune dead wood, sanitize pruning tools.",
        "treatment_timeline": "Apply 3-4 sprays at 10-day intervals"
    },
    "Apple___Cedar_apple_rust": {
        "cause": "Fungal disease caused by Gymnosporangium juniperi-virginianae.",
        "pesticide": "Myclobutanil or Propiconazole",
        "dosage": "1ml per litre of water",
        "prevention": "Remove nearby juniper trees if possible, apply fungicide before infection period.",
        "treatment_timeline": "Apply from pink bud stage through 3rd cover spray"
    },
    "Cherry_(including_sour)___Powdery_mildew": {
        "cause": "Fungal infection caused by Podosphaera clandestina.",
        "pesticide": "Sulfur-based fungicide or Trifloxystrobin",
        "dosage": "3g per litre of water",
        "prevention": "Avoid excessive nitrogen fertilization, ensure good air circulation.",
        "treatment_timeline": "Apply every 10-14 days from bud break"
    },
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "cause": "Fungal disease caused by Cercospora zeae-maydis, favored by warm humid weather.",
        "pesticide": "Azoxystrobin or Propiconazole",
        "dosage": "1.5ml per litre of water",
        "prevention": "Crop rotation, use resistant hybrids, avoid surface irrigation.",
        "treatment_timeline": "Apply at first sign of disease, repeat in 14 days if needed"
    },
    "Corn_(maize)___Common_rust_": {
        "cause": "Caused by Puccinia sorghi fungus, spreads via wind-blown spores.",
        "pesticide": "Mancozeb or Propiconazole",
        "dosage": "2.5g per litre of water",
        "prevention": "Plant resistant varieties, early planting to avoid peak spore periods.",
        "treatment_timeline": "Apply fungicide at early rust detection, before 50% infection"
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "cause": "Caused by Exserohilum turcicum fungus.",
        "pesticide": "Azoxystrobin + Propiconazole (combo)",
        "dosage": "2ml per litre of water",
        "prevention": "Crop rotation with non-host crops, use resistant hybrids.",
        "treatment_timeline": "Single application at early tassel stage is usually sufficient"
    },
    "Grape___Black_rot": {
        "cause": "Fungal disease caused by Guignardia bidwellii.",
        "pesticide": "Mancozeb or Myclobutanil",
        "dosage": "2.5g per litre of water",
        "prevention": "Remove infected berries and leaves, improve canopy airflow.",
        "treatment_timeline": "Begin sprays at bud break, every 10-14 days through veraison"
    },
    "Grape___Esca_(Black_Measles)": {
        "cause": "Complex fungal disease involving Phaeomoniella chlamydospora and others.",
        "pesticide": "No curative fungicide available — consult local agricultural authority",
        "dosage": "Consult local agricultural authority",
        "prevention": "Protect pruning wounds, remove and destroy infected wood.",
        "treatment_timeline": "Preventive wound protection at every pruning"
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "cause": "Caused by Isariopsis clavispora fungus.",
        "pesticide": "Copper oxychloride or Mancozeb",
        "dosage": "3g per litre of water",
        "prevention": "Avoid wetting foliage, improve air circulation.",
        "treatment_timeline": "Spray every 10 days during humid conditions"
    },
    "Orange___Haunglongbing_(Citrus_greening)": {
        "cause": "Bacterial disease caused by Candidatus Liberibacter, spread by Asian citrus psyllid.",
        "pesticide": "Control psyllid vector with Imidacloprid or Thiamethoxam",
        "dosage": "0.5ml per litre for foliar spray",
        "prevention": "Remove infected trees immediately, control psyllid population aggressively.",
        "treatment_timeline": "Ongoing — currently no cure, focus on vector control"
    },
    "Peach___Bacterial_spot": {
        "cause": "Caused by Xanthomonas arboricola pv. pruni bacterium.",
        "pesticide": "Copper-based bactericide (Copper hydroxide)",
        "dosage": "3g per litre of water",
        "prevention": "Plant resistant varieties, avoid overhead irrigation.",
        "treatment_timeline": "Apply from shuck split through 2nd cover, every 7-10 days"
    },
    "Pepper,_bell___Bacterial_spot": {
        "cause": "Caused by Xanthomonas campestris pv. vesicatoria.",
        "pesticide": "Copper hydroxide + Mancozeb combo",
        "dosage": "2.5g per litre of water",
        "prevention": "Use disease-free seeds, avoid overhead watering.",
        "treatment_timeline": "Weekly sprays during warm/wet weather"
    },
    "Potato___Early_blight": {
        "cause": "Caused by Alternaria solani fungus, attacks older leaves first.",
        "pesticide": "Mancozeb 75 WP or Chlorothalonil",
        "dosage": "2.5g per litre of water",
        "prevention": "Proper crop rotation, avoid excessive nitrogen, remove infected debris.",
        "treatment_timeline": "Apply every 7-10 days starting at first sign"
    },
    "Potato___Late_blight": {
        "cause": "Caused by Phytophthora infestans — extremely destructive in cool wet weather.",
        "pesticide": "Metalaxyl + Mancozeb (Ridomil Gold) or Cymoxanil",
        "dosage": "2.5g per litre of water",
        "prevention": "Plant certified disease-free seed, avoid overhead irrigation, destroy volunteer plants.",
        "treatment_timeline": "Preventive sprays every 7 days in cool/wet weather — act fast, spreads rapidly"
    },
    "Squash___Powdery_mildew": {
        "cause": "Caused by Podosphaera xanthii or Erysiphe cichoracearum fungi.",
        "pesticide": "Potassium bicarbonate or Sulfur-based fungicide",
        "dosage": "5g per litre of water",
        "prevention": "Plant resistant varieties, avoid overhead watering.",
        "treatment_timeline": "Apply every 7-14 days from first sign"
    },
    "Strawberry___Leaf_scorch": {
        "cause": "Caused by Diplocarpon earlianum fungus.",
        "pesticide": "Captan or Myclobutanil",
        "dosage": "2g per litre of water",
        "prevention": "Remove old infected leaves after harvest, avoid wet foliage.",
        "treatment_timeline": "Apply in early spring and again after renovation"
    },
    "Tomato___Bacterial_spot": {
        "cause": "Caused by Xanthomonas species bacteria.",
        "pesticide": "Copper hydroxide or Copper oxychloride",
        "dosage": "3g per litre of water",
        "prevention": "Use disease-free transplants, stake plants for airflow.",
        "treatment_timeline": "Weekly sprays during warm humid weather"
    },
    "Tomato___Early_blight": {
        "cause": "Caused by Alternaria solani fungus.",
        "pesticide": "Mancozeb or Chlorothalonil",
        "dosage": "2.5g per litre of water",
        "prevention": "Mulch around plants, remove lower infected leaves, rotate crops.",
        "treatment_timeline": "Every 7-10 days from first sign"
    },
    "Tomato___Late_blight": {
        "cause": "Caused by Phytophthora infestans. Extremely destructive in cool wet weather.",
        "pesticide": "Metalaxyl + Mancozeb or Cymoxanil + Mancozeb",
        "dosage": "2.5g per litre of water",
        "prevention": "Avoid overhead watering, remove infected plants immediately.",
        "treatment_timeline": "Every 5-7 days during high-risk weather — do not delay"
    },
    "Tomato___Leaf_Mold": {
        "cause": "Caused by Passalora fulva fungus, thrives in high humidity.",
        "pesticide": "Chlorothalonil or Mancozeb",
        "dosage": "2g per litre of water",
        "prevention": "Reduce humidity in greenhouse, improve ventilation.",
        "treatment_timeline": "Apply every 10-14 days"
    },
    "Tomato___Septoria_leaf_spot": {
        "cause": "Caused by Septoria lycopersici fungus.",
        "pesticide": "Mancozeb or Copper-based fungicide",
        "dosage": "2.5g per litre of water",
        "prevention": "Avoid wetting leaves, mulch soil to prevent splash spread.",
        "treatment_timeline": "Apply every 10 days starting from first sign"
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "cause": "Infestation by Tetranychus urticae — a pest, not a fungal disease.",
        "pesticide": "Abamectin or Spiromesifen (miticide)",
        "dosage": "1ml per litre of water",
        "prevention": "Avoid water stress, use predatory mites as biological control.",
        "treatment_timeline": "Apply 2-3 times at 5-7 day intervals"
    },
    "Tomato___Target_Spot": {
        "cause": "Caused by Corynespora cassiicola fungus.",
        "pesticide": "Azoxystrobin or Difenoconazole",
        "dosage": "1ml per litre of water",
        "prevention": "Improve air circulation, stake plants, avoid leaf wetness.",
        "treatment_timeline": "Apply every 7-14 days"
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "cause": "Viral disease spread by whitefly (Bemisia tabaci).",
        "pesticide": "Control whitefly with Imidacloprid or Thiamethoxam",
        "dosage": "0.5ml per litre of water",
        "prevention": "Use reflective mulch, yellow sticky traps, remove infected plants.",
        "treatment_timeline": "Ongoing vector control — no cure for infected plants"
    },
    "Tomato___Tomato_mosaic_virus": {
        "cause": "Viral disease spread by contact, tools, and insects.",
        "pesticide": "No chemical cure — remove infected plants immediately",
        "dosage": "N/A",
        "prevention": "Sanitize tools, wash hands, use virus-resistant varieties.",
        "treatment_timeline": "Preventive only — remove infected plants immediately"
    },
}

DEFAULT_RECOMMENDATION = {
    "cause": "Specific cause data not available. Consult your local agricultural extension officer.",
    "pesticide": "Consult local agricultural authority",
    "dosage": "As per label instructions",
    "prevention": "Practice crop rotation, proper irrigation, and regular field scouting.",
    "treatment_timeline": "Apply treatment as advised by agricultural expert"
}

_disease_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


# ─────────────────────────────────────────────────────────────────
# Internal loaders — called once, cached globally
# ─────────────────────────────────────────────────────────────────

def _load_yield_models():
    global _yield_model, _encoder_crop, _encoder_season, _encoder_state
    if _yield_model is None:
        print("📦 Loading yield model...")
        _yield_model    = joblib.load(YIELD_MODEL_PATH)
        _encoder_crop   = joblib.load(ENCODER_CROP_PATH)
        _encoder_season = joblib.load(ENCODER_SEASON_PATH)
        _encoder_state  = joblib.load(ENCODER_STATE_PATH)
        print("✅ Yield model loaded.")


def _load_crop_rec_models():
    global _crop_rec_model, _crop_rec_scaler, _crop_rec_encoder
    if _crop_rec_model is None:
        print("📦 Loading crop recommendation model...")
        _crop_rec_model   = joblib.load(CROP_REC_MODEL_PATH)
        _crop_rec_scaler  = joblib.load(CROP_REC_SCALER_PATH)
        _crop_rec_encoder = joblib.load(CROP_REC_ENCODER_PATH)
        print("✅ Crop recommendation model loaded.")


def _load_disease_model():
    global _disease_model
    if _disease_model is not None:
        return _disease_model

    print("🌿 Loading plant disease model...")
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, len(CLASS_NAMES))

    if not os.path.exists(DISEASE_MODEL_PATH):
        print("📥 Downloading pretrained weights from HuggingFace (one-time ~50MB)...")
        try:
            from huggingface_hub import hf_hub_download
            import shutil
            os.makedirs(MODELS_DIR, exist_ok=True)
            weights_path = hf_hub_download(
                repo_id="linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification",
                filename="pytorch_model.bin"
            )
            shutil.copy(weights_path, DISEASE_MODEL_PATH)
            print("✅ Weights downloaded and cached.")
        except Exception as e:
            raise RuntimeError(f"Failed to download disease model weights: {e}")

    state_dict = torch.load(DISEASE_MODEL_PATH, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    _disease_model = model
    print("✅ Plant disease model loaded.")
    return _disease_model


# ─────────────────────────────────────────────────────────────────
# 1. YIELD PREDICTION
# ─────────────────────────────────────────────────────────────────

def predict_yield(
    crop: str,
    crop_year: int,
    season: str,
    state: str,
    area: float,
    annual_rainfall: float,
    fertilizer: float,
    pesticide: float
) -> dict:
    """
    Predict crop yield in tons/hectare.

    Args:
        crop           : Crop name e.g. "Rice"
        crop_year      : Year e.g. 2024
        season         : Season e.g. "Kharif", "Rabi", "Whole Year"
        state          : Indian state e.g. "Karnataka"
        area           : Area under cultivation in hectares
        annual_rainfall: Annual rainfall in mm
        fertilizer     : Fertilizer used in kg
        pesticide      : Pesticide used in kg

    Returns:
        { "predicted_yield": float, "unit": "tons/hectare", "input_summary": dict }
    """
    _load_yield_models()

    # Validate and encode categoricals
    try:
        crop_enc   = _encoder_crop.transform([crop])[0]
    except ValueError:
        raise ValueError(f"Unknown crop '{crop}'. Use /api/yield-options to get valid values.")

    try:
        season_enc = _encoder_season.transform([season])[0]
    except ValueError:
        raise ValueError(f"Unknown season '{season}'. Use /api/yield-options to get valid values.")

    try:
        state_enc  = _encoder_state.transform([state])[0]
    except ValueError:
        raise ValueError(f"Unknown state '{state}'. Use /api/yield-options to get valid values.")

    features = np.array([[
        crop_enc,
        crop_year,
        season_enc,
        state_enc,
        area,
        annual_rainfall,
        fertilizer,
        pesticide
    ]])

    predicted = float(_yield_model.predict(features)[0])
    predicted = round(max(predicted, 0), 4)  # no negative yield

    return {
        "predicted_yield": predicted,
        "unit": "tons/hectare",
        "input_summary": {
            "crop": crop,
            "year": crop_year,
            "season": season,
            "state": state,
            "area_hectares": area,
            "annual_rainfall_mm": annual_rainfall,
            "fertilizer_kg": fertilizer,
            "pesticide_kg": pesticide
        }
    }


def get_yield_options() -> dict:
    """Returns valid values for crop, season, and state dropdowns."""
    _load_yield_models()
    return {
        "crops":   sorted(_encoder_crop.classes_.tolist()),
        "seasons": sorted(_encoder_season.classes_.tolist()),
        "states":  sorted(_encoder_state.classes_.tolist())
    }


# ─────────────────────────────────────────────────────────────────
# 2. CROP RECOMMENDATION
# ─────────────────────────────────────────────────────────────────

def recommend_crop(
    N: float,
    P: float,
    K: float,
    temperature: float,
    humidity: float,
    ph: float,
    rainfall: float
) -> dict:
    """
    Recommend the best crop based on soil and weather conditions.

    Args:
        N           : Nitrogen content in soil (kg/ha)
        P           : Phosphorus content in soil (kg/ha)
        K           : Potassium content in soil (kg/ha)
        temperature : Temperature in Celsius
        humidity    : Relative humidity in %
        ph          : Soil pH value
        rainfall    : Rainfall in mm

    Returns:
        {
            "recommended_crop": str,
            "confidence": float,
            "top_3": [{ "crop": str, "confidence": float }]
        }
    """
    _load_crop_rec_models()

    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    features_scaled = _crop_rec_scaler.transform(features)

    probabilities = _crop_rec_model.predict_proba(features_scaled)[0]
    top_indices = np.argsort(probabilities)[::-1][:3]

    recommended_crop = _crop_rec_encoder.inverse_transform([top_indices[0]])[0]
    confidence = round(float(probabilities[top_indices[0]]) * 100, 2)

    top_3 = [
        {
            "crop": _crop_rec_encoder.inverse_transform([idx])[0],
            "confidence": round(float(probabilities[idx]) * 100, 2)
        }
        for idx in top_indices
    ]

    return {
        "recommended_crop": recommended_crop,
        "confidence": confidence,
        "top_3": top_3,
        "input_summary": {
            "N": N, "P": P, "K": K,
            "temperature": temperature,
            "humidity": humidity,
            "ph": ph,
            "rainfall": rainfall
        }
    }


# ─────────────────────────────────────────────────────────────────
# 3. DISEASE DETECTION
# ─────────────────────────────────────────────────────────────────

def detect_disease(image_bytes: bytes) -> dict:
    """
    Detect plant disease from a leaf image.

    Args:
        image_bytes: Raw image bytes from UploadFile.read()

    Returns:
        {
            "disease": str,
            "crop": str,
            "is_healthy": bool,
            "confidence": float,
            "cause": str,
            "pesticide": str,
            "dosage": str,
            "prevention": str,
            "treatment_timeline": str,
            "top_3": [{ "disease": str, "confidence": float }]
        }
    """
    model = _load_disease_model()

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = _disease_transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

    top_prob, top_idx = torch.topk(probabilities, 3)
    top_probs   = top_prob.tolist()
    top_indices = top_idx.tolist()

    predicted_class = CLASS_NAMES[top_indices[0]]
    confidence      = round(top_probs[0] * 100, 2)

    parts      = predicted_class.split("___")
    crop       = parts[0].replace("_", " ").replace("(including sour)", "").strip()
    disease    = parts[1].replace("_", " ").strip() if len(parts) > 1 else "Unknown"
    is_healthy = "healthy" in predicted_class.lower()

    rec = DISEASE_RECOMMENDATIONS.get(predicted_class, DEFAULT_RECOMMENDATION)

    top_3 = []
    for i in range(3):
        cls         = CLASS_NAMES[top_indices[i]]
        cls_parts   = cls.split("___")
        cls_disease = cls_parts[1].replace("_", " ").strip() if len(cls_parts) > 1 else cls
        top_3.append({
            "disease":    cls_disease,
            "confidence": round(top_probs[i] * 100, 2)
        })

    return {
        "disease":            disease,
        "crop":               crop,
        "is_healthy":         is_healthy,
        "confidence":         confidence,
        "cause":              rec["cause"],
        "pesticide":          rec["pesticide"],
        "dosage":             rec["dosage"],
        "prevention":         rec["prevention"],
        "treatment_timeline": rec["treatment_timeline"],
        "top_3":              top_3
    }


def get_supported_crops() -> list:
    """Returns list of crops supported by the disease detection model."""
    crops = set()
    for cls in CLASS_NAMES:
        crop = cls.split("___")[0].replace("_", " ").strip()
        crops.add(crop)
    return sorted(list(crops))