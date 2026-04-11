"""
routes/soil.py
--------------
Soil profile endpoints — protected routes.
Farmer enters soil data once, reused for all crop recommendations.

  POST /api/soil-profile  → save or update soil data
  GET  /api/soil-profile  → fetch saved soil data
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from routes.auth      import get_current_farmer
from supabase_service import save_soil_profile, get_soil_profile

router = APIRouter()


class SoilProfileRequest(BaseModel):
    N:  float = Field(..., description="Nitrogen in soil (kg/ha)",   example=90)
    P:  float = Field(..., description="Phosphorus in soil (kg/ha)", example=42)
    K:  float = Field(..., description="Potassium in soil (kg/ha)",  example=43)
    ph: float = Field(..., description="Soil pH value",              example=6.5)


@router.post("/soil-profile")
def save_soil(
    req: SoilProfileRequest,
    farmer: dict = Depends(get_current_farmer)
):
    """
    Save or update farmer's soil profile.
    Call this once after soil testing. Updates automatically if called again.
    """
    try:
        profile = save_soil_profile(
            farmer_id=farmer["id"],
            n=req.N,
            p=req.P,
            k=req.K,
            ph=req.ph,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save soil profile: {str(e)}")

    return {
        "success": True,
        "message": "Soil profile saved successfully.",
        "data": {
            "N":         profile["n"],
            "P":         profile["p"],
            "K":         profile["k"],
            "ph":        profile["ph"],
            "updated_at": profile["updated_at"],
        }
    }


@router.get("/soil-profile")
def fetch_soil(farmer: dict = Depends(get_current_farmer)):
    """
    Fetch farmer's saved soil profile.
    Returns null if farmer hasn't set up soil profile yet.
    """
    profile = get_soil_profile(farmer["id"])

    if not profile:
        return {
            "success": True,
            "data":    None,
            "message": "No soil profile found. Please set up your soil profile first."
        }

    return {
        "success": True,
        "data": {
            "N":          profile["n"],
            "P":          profile["p"],
            "K":          profile["k"],
            "ph":         profile["ph"],
            "updated_at": profile["updated_at"],
        }
    }