"""
routes/disease.py
-----------------
Plant disease detection — protected route.
Requires JWT token. Uploads image to Supabase Storage. Saves result to DB.
"""

import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from routes.auth         import get_current_farmer
from ml_service          import detect_disease, get_supported_crops
from supabase_service    import save_disease_detection, get_disease_history, upload_disease_image , get_disease_by_id

router = APIRouter()

ALLOWED_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
MAX_SIZE_MB   = 10


@router.post("/detect-disease")
async def detect(
    image: UploadFile = File(...),
    farmer: dict = Depends(get_current_farmer)
):
    """
    Upload a plant leaf image to detect disease.
    Supports: jpg, jpeg, png, webp. Max size: 10MB.
    """
    # Validate file type
    if image.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{image.content_type}'. Use jpg, png, or webp."
        )

    image_bytes = await image.read()

    # Validate file size
    if len(image_bytes) > MAX_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail=f"Image too large. Maximum size is {MAX_SIZE_MB}MB."
        )

    # Step 1: Run ML model
    try:
        result = detect_disease(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

    # Step 2: Upload image to Supabase Storage
    image_url = ""
    try:
        ext        = image.content_type.split("/")[-1]
        filename   = f"{uuid.uuid4()}.{ext}"
        image_url  = upload_disease_image(farmer["id"], image_bytes, filename)
    except Exception:
        pass  # Don't fail if image upload fails — result still gets returned

    # Step 3: Save to DB
    try:
        save_disease_detection(
            farmer_id=farmer["id"],
            image_url=image_url,
            crop=result["crop"],
            disease=result["disease"],
            is_healthy=result["is_healthy"],
            confidence=result["confidence"],
            cause=result["cause"],
            pesticide=result["pesticide"],
            dosage=result["dosage"],
            prevention=result["prevention"],
            treatment_timeline=result["treatment_timeline"],
            top_3=result["top_3"],
        )
    except Exception:
        pass  # Don't fail the request if DB save fails

    return {
        "success": True,
        "data": {
            **result,
            "image_url": image_url,
        }
    }


@router.get("/disease-history")
def history(
    limit: int = 10,
    farmer: dict = Depends(get_current_farmer)
):
    """Returns the farmer's last N disease detections."""
    records = get_disease_history(farmer["id"], limit=limit)
    return {"success": True, "data": records}

@router.get("/disease-history/{detection_id}")
def get_single_disease(         # ← different from get_disease_by_id
    detection_id: str,
    farmer: dict = Depends(get_current_farmer)
):
    result = get_disease_by_id(farmer["id"], detection_id)
    if not result:
        raise HTTPException(status_code=404, detail="Detection not found.")
    return {"success": True, "data": result}

@router.get("/supported-crops")
def supported_crops():
    """Returns all crops supported by the disease model. Public endpoint."""
    return {
        "crops": get_supported_crops(),
        "total_disease_classes": 38,
    }