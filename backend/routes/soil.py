"""
routes/soil.py
--------------
Soil profile endpoints — protected routes.
Farmer enters soil data once, reused for all crop recommendations.

  POST  /api/soil-profile  → save soil data (first time)
  GET   /api/soil-profile  → fetch saved soil data
  PATCH /api/soil-profile  → partially update soil data (after re-testing)
"""

from typing import Optional
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


class SoilProfileUpdate(BaseModel):
    N:  Optional[float] = Field(None, description="Nitrogen in soil (kg/ha)",   example=95)
    P:  Optional[float] = Field(None, description="Phosphorus in soil (kg/ha)", example=45)
    K:  Optional[float] = Field(None, description="Potassium in soil (kg/ha)",  example=40)
    ph: Optional[float] = Field(None, description="Soil pH value",              example=6.8)


@router.post("/soil-profile")
def save_soil(
    req: SoilProfileRequest,
    farmer: dict = Depends(get_current_farmer)
):
    """
    Save farmer's soil profile for the first time.
    Call this once after soil testing.
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
            "N":          profile["n"],
            "P":          profile["p"],
            "K":          profile["k"],
            "ph":         profile["ph"],
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


@router.patch("/soil-profile")
def update_soil(
    req: SoilProfileUpdate,
    farmer: dict = Depends(get_current_farmer)
):
    """
    Partially update soil profile.
    Only provide the values you want to change.
    Useful when farmer re-tests soil seasonally.

    Example — update only pH:
        { "ph": 7.0 }

    Example — update all after full soil test:
        { "N": 95, "P": 45, "K": 40, "ph": 6.8 }
    """
    # Load existing profile first
    soil = get_soil_profile(farmer["id"])
    if not soil:
        raise HTTPException(
            status_code=404,
            detail="No soil profile found. Please create one first via POST /api/soil-profile."
        )

    # Only update fields that were provided, keep existing values for the rest
    updated_n  = req.N  if req.N  is not None else soil["n"]
    updated_p  = req.P  if req.P  is not None else soil["p"]
    updated_k  = req.K  if req.K  is not None else soil["k"]
    updated_ph = req.ph if req.ph is not None else soil["ph"]

    try:
        profile = save_soil_profile(
            farmer_id=farmer["id"],
            n=updated_n,
            p=updated_p,
            k=updated_k,
            ph=updated_ph,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update soil profile: {str(e)}")

    return {
        "success": True,
        "message": "Soil profile updated successfully.",
        "data": {
            "N":          profile["n"],
            "P":          profile["p"],
            "K":          profile["k"],
            "ph":         profile["ph"],
            "updated_at": profile["updated_at"],
        }
    }