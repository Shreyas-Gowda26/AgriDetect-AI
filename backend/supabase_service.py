"""
supabase_service.py
-------------------
All Supabase DB operations for AgriPredict.
Handles farmers auth + saving all three prediction histories.

Add to .env:
    SUPABASE_URL=https://your-project.supabase.co
    SUPABASE_SERVICE_KEY=your-service-role-key   ← NOT the anon key
"""

import os
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL         = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    raise RuntimeError(
        "SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in your .env file."
    )

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)


# ──────────────────────────────────────────────
# FARMERS — Auth
# ──────────────────────────────────────────────

def create_farmer(full_name: str, phone: str, password_hash: str, city: str, state: str) -> dict:
    """Register a new farmer. Raises if phone already exists."""
    result = supabase.table("farmers").insert({
        "full_name":     full_name,
        "phone":         phone,
        "password_hash": password_hash,
        "city":          city,
        "state":         state,
    }).execute()

    if not result.data:
        raise ValueError("Failed to create farmer account.")
    return result.data[0]


def get_farmer_by_phone(phone: str) -> dict | None:
    """Fetch farmer by phone number. Returns None if not found."""
    result = supabase.table("farmers") \
        .select("*") \
        .eq("phone", phone) \
        .limit(1) \
        .execute()
    return result.data[0] if result.data else None


def get_farmer_by_id(farmer_id: str) -> dict | None:
    """Fetch farmer by UUID. Returns None if not found."""
    result = supabase.table("farmers") \
        .select("id, full_name, phone, city, state, created_at") \
        .eq("id", farmer_id) \
        .limit(1) \
        .execute()
    return result.data[0] if result.data else None


# ──────────────────────────────────────────────
# YIELD PREDICTIONS — Save & Fetch history
# ──────────────────────────────────────────────

def save_yield_prediction(
    farmer_id: str,
    crop: str,
    crop_year: int,
    season: str,
    state: str,
    area: float,
    fertilizer: float,
    pesticide: float,
    city: str,
    annual_rainfall: float,
    predicted_yield: float,
) -> dict:
    """Save a yield prediction result to DB."""
    result = supabase.table("yield_predictions").insert({
        "farmer_id":      farmer_id,
        "crop":           crop,
        "crop_year":      crop_year,
        "season":         season,
        "state":          state,
        "area":           area,
        "fertilizer":     fertilizer,
        "pesticide":      pesticide,
        "city":           city,
        "annual_rainfall": annual_rainfall,
        "predicted_yield": predicted_yield,
    }).execute()

    if not result.data:
        raise ValueError("Failed to save yield prediction.")
    return result.data[0]


def get_yield_history(farmer_id: str, limit: int = 10) -> list:
    """Fetch the last N yield predictions for a farmer."""
    result = supabase.table("yield_predictions") \
        .select("*") \
        .eq("farmer_id", farmer_id) \
        .order("created_at", desc=True) \
        .limit(limit) \
        .execute()
    return result.data or []


# ──────────────────────────────────────────────
# CROP RECOMMENDATIONS — Save & Fetch history
# ──────────────────────────────────────────────

def save_crop_recommendation(
    farmer_id: str,
    n: float, p: float, k: float, ph: float,
    city: str,
    temperature: float,
    humidity: float,
    rainfall: float,
    recommended_crop: str,
    confidence: float,
    top_3: list,
) -> dict:
    """Save a crop recommendation result to DB."""
    result = supabase.table("crop_recommendations").insert({
        "farmer_id":        farmer_id,
        "n":                n,
        "p":                p,
        "k":                k,
        "ph":               ph,
        "city":             city,
        "temperature":      temperature,
        "humidity":         humidity,
        "rainfall":         rainfall,
        "recommended_crop": recommended_crop,
        "confidence":       confidence,
        "top_3":            top_3,
    }).execute()

    if not result.data:
        raise ValueError("Failed to save crop recommendation.")
    return result.data[0]


def get_crop_rec_history(farmer_id: str, limit: int = 10) -> list:
    """Fetch the last N crop recommendations for a farmer."""
    result = supabase.table("crop_recommendations") \
        .select("*") \
        .eq("farmer_id", farmer_id) \
        .order("created_at", desc=True) \
        .limit(limit) \
        .execute()
    return result.data or []


# ──────────────────────────────────────────────
# DISEASE DETECTIONS — Save & Fetch history
# ──────────────────────────────────────────────

def save_disease_detection(
    farmer_id: str,
    image_url: str,
    crop: str,
    disease: str,
    is_healthy: bool,
    confidence: float,
    cause: str,
    pesticide: str,
    dosage: str,
    prevention: str,
    treatment_timeline: str,
    top_3: list,
) -> dict:
    """Save a disease detection result to DB."""
    result = supabase.table("disease_detections").insert({
        "farmer_id":          farmer_id,
        "image_url":          image_url,
        "crop":               crop,
        "disease":            disease,
        "is_healthy":         is_healthy,
        "confidence":         confidence,
        "cause":              cause,
        "pesticide":          pesticide,
        "dosage":             dosage,
        "prevention":         prevention,
        "treatment_timeline": treatment_timeline,
        "top_3":              top_3,
    }).execute()

    if not result.data:
        raise ValueError("Failed to save disease detection.")
    return result.data[0]


def get_disease_history(farmer_id: str, limit: int = 10) -> list:
    """Fetch the last N disease detections for a farmer."""
    result = supabase.table("disease_detections") \
        .select("*") \
        .eq("farmer_id", farmer_id) \
        .order("created_at", desc=True) \
        .limit(limit) \
        .execute()
    return result.data or []


# ──────────────────────────────────────────────
# STORAGE — Upload disease image
# ──────────────────────────────────────────────

def upload_disease_image(farmer_id: str, image_bytes: bytes, filename: str) -> str:
    """
    Upload a leaf image to Supabase Storage.
    Images are stored under disease-images/{farmer_id}/{filename}
    Returns the public URL of the uploaded image.
    """
    storage_path = f"{farmer_id}/{filename}"

    supabase.storage.from_("disease-images").upload(
        path=storage_path,
        file=image_bytes,
        file_options={"content-type": "image/jpeg", "upsert": "true"}
    )

    # Get signed URL valid for 1 year (since bucket is private)
    result = supabase.storage.from_("disease-images").create_signed_url(
        path=storage_path,
        expires_in=365 * 24 * 3600
    )
    return result["signedURL"]


# ──────────────────────────────────────────────
# SOIL PROFILE — Save & Fetch
# ──────────────────────────────────────────────

def save_soil_profile(farmer_id: str, n: float, p: float, k: float, ph: float) -> dict:
    """
    Save or update soil profile for a farmer.
    Uses upsert so it creates on first save, updates on subsequent saves.
    """
    result = supabase.table("soil_profiles").upsert({
        "farmer_id":  farmer_id,
        "n":          n,
        "p":          p,
        "k":          k,
        "ph":         ph,
        "updated_at": "now()",
    }, on_conflict="farmer_id").execute()

    if not result.data:
        raise ValueError("Failed to save soil profile.")
    return result.data[0]


def get_soil_profile(farmer_id: str) -> dict | None:
    """Fetch soil profile for a farmer. Returns None if not set yet."""
    result = supabase.table("soil_profiles") \
        .select("*") \
        .eq("farmer_id", farmer_id) \
        .limit(1) \
        .execute()
    return result.data[0] if result.data else None

def get_yield_by_id(farmer_id: str, prediction_id: str) -> dict | None:
    result = supabase.table("yield_predictions") \
        .select("*") \
        .eq("id", prediction_id) \
        .eq("farmer_id", farmer_id) \
        .limit(1) \
        .execute()
    return result.data[0] if result.data else None


def get_crop_rec_by_id(farmer_id: str, recommendation_id: str) -> dict | None:
    result = supabase.table("crop_recommendations") \
        .select("*") \
        .eq("id", recommendation_id) \
        .eq("farmer_id", farmer_id) \
        .limit(1) \
        .execute()
    return result.data[0] if result.data else None


def get_disease_by_id(farmer_id: str, detection_id: str) -> dict | None:
    result = supabase.table("disease_detections") \
        .select("*") \
        .eq("id", detection_id) \
        .eq("farmer_id", farmer_id) \
        .limit(1) \
        .execute()
    return result.data[0] if result.data else None