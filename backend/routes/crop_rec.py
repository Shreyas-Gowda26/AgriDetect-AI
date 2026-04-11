"""
routes/crop_rec.py
------------------
Crop recommendation — protected route.
Farmer only provides city name.
Soil data auto-loaded from saved profile.
Weather auto-fetched from OpenWeatherMap.
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from routes.auth         import get_current_farmer
from ml_service          import recommend_crop
from weather_service     import get_weather_by_city
from supabase_service    import save_crop_recommendation, get_crop_rec_history, get_soil_profile,get_crop_rec_by_id

router = APIRouter()


class CropRecRequest(BaseModel):
    city: str = Field(..., example="Mysuru")


@router.post("/recommend-crop")
async def recommend(
    req: CropRecRequest,
    farmer: dict = Depends(get_current_farmer)
):
    """
    Recommend the best crop to grow.
    Farmer only needs to provide city name.
    Soil data auto-loaded from saved profile.
    Weather auto-fetched from OpenWeatherMap.
    """
    # Step 1: Load soil profile
    soil = get_soil_profile(farmer["id"])
    if not soil:
        raise HTTPException(
            status_code=400,
            detail="No soil profile found. Please set up your soil profile first via POST /api/soil-profile."
        )
    N  = soil["n"]
    P  = soil["p"]
    K  = soil["k"]
    ph = soil["ph"]

    # Step 2: Fetch weather
    try:
        weather = await get_weather_by_city(req.city)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    # Step 3: Run ML model
    try:
        result = recommend_crop(
            N=N, P=P, K=K,
            temperature=weather["temperature"],
            humidity=weather["humidity"],
            ph=ph,
            rainfall=weather["annual_rainfall_estimate"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    # Step 4: Save to DB
    try:
        save_crop_recommendation(
            farmer_id=farmer["id"],
            n=N, p=P, k=K, ph=ph,
            city=weather["city"],
            temperature=weather["temperature"],
            humidity=weather["humidity"],
            rainfall=weather["annual_rainfall_estimate"],
            recommended_crop=result["recommended_crop"],
            confidence=result["confidence"],
            top_3=result["top_3"],
        )
    except Exception as e:
        print(f"❌ DB SAVE ERROR: {e}")

    return {
        "success": True,
        "data": {
            **result,
            "soil_used": {
                "N": N, "P": P, "K": K, "ph": ph,
                "source": "soil_profile"
            },
            "weather_used": {
                "city":        weather["city"],
                "temperature": weather["temperature"],
                "humidity":    weather["humidity"],
                "rainfall_mm": weather["annual_rainfall_estimate"],
                "condition":   weather["weather_desc"],
            }
        }
    }


@router.get("/crop-rec-history")
def history(
    limit: int = 10,
    farmer: dict = Depends(get_current_farmer)
):
    """Returns the farmer's last N crop recommendations."""
    records = get_crop_rec_history(farmer["id"], limit=limit)
    return {"success": True, "data": records}

@router.get("/crop-rec-history/{recommendation_id}")
def get_single_crop_rec(        # ← different from get_crop_rec_by_id
    recommendation_id: str,
    farmer: dict = Depends(get_current_farmer)
):
    result = get_crop_rec_by_id(farmer["id"], recommendation_id)
    if not result:
        raise HTTPException(status_code=404, detail="Recommendation not found.")
    return {"success": True, "data": result}

@router.get("/crop-rec-options")
def options():
    """Valid input ranges for crop recommendation. Public endpoint."""
    return {
        "N":    {"min": 0,   "max": 140, "unit": "kg/ha"},
        "P":    {"min": 5,   "max": 145, "unit": "kg/ha"},
        "K":    {"min": 5,   "max": 205, "unit": "kg/ha"},
        "ph":   {"min": 3.5, "max": 9.9},
        "note": "Temperature, humidity, rainfall auto-fetched. Soil values auto-loaded from profile."
    }