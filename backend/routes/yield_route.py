"""
routes/yield_route.py
---------------------
Crop yield prediction — protected route.
Requires JWT token. Auto-fetches weather. Saves result to DB.
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from routes.auth         import get_current_farmer
from ml_service          import predict_yield, get_yield_options
from weather_service     import get_weather_by_city
from supabase_service    import save_yield_prediction, get_yield_history,get_yield_by_id

router = APIRouter()


class YieldRequest(BaseModel):
    city:       str   = Field(..., example="Mysuru")
    crop:       str   = Field(..., example="Rice")
    crop_year:  int   = Field(..., example=2024)
    season:     str   = Field(..., example="Kharif")
    state:      str   = Field(None, example="Karnataka",
                              description="Optional — auto-detected from city if not provided")
    area:       float = Field(..., example=5000,   description="Area in hectares")
    fertilizer: float = Field(..., example=500000, description="Fertilizer used in kg")
    pesticide:  float = Field(..., example=15000,  description="Pesticide used in kg")


@router.post("/predict-yield")
async def predict(
    req: YieldRequest,
    farmer: dict = Depends(get_current_farmer)
):
    # Step 1: Fetch weather
    try:
        weather = await get_weather_by_city(req.city)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    # Step 2: Resolve state
    state = req.state or weather.get("state", "")
    if not state:
        raise HTTPException(
            status_code=400,
            detail=f"Could not auto-detect state for '{req.city}'. Please provide 'state' manually."
        )

    # Step 3: Run ML model
    try:
        result = predict_yield(
            crop=req.crop,
            crop_year=req.crop_year,
            season=req.season,
            state=state,
            area=req.area,
            annual_rainfall=weather["annual_rainfall_estimate"],
            fertilizer=req.fertilizer,
            pesticide=req.pesticide,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    # Step 4: Save to DB
    try:
        save_yield_prediction(
            farmer_id=farmer["id"],
            crop=req.crop,
            crop_year=req.crop_year,
            season=req.season,
            state=state,
            area=req.area,
            fertilizer=req.fertilizer,
            pesticide=req.pesticide,
            city=weather["city"],
            annual_rainfall=weather["annual_rainfall_estimate"],
            predicted_yield=result["predicted_yield"],
        )
    except Exception:
        print(f"❌ DB SAVE ERROR: {e}") 

    return {
        "success": True,
        "data": {
            **result,
            "weather_used": {
                "city":               weather["city"],
                "annual_rainfall_mm": weather["annual_rainfall_estimate"],
                "condition":          weather["weather_desc"],
            },
            "state_used": state,
        }
    }


@router.get("/yield-history")
def history(
    limit: int = 10,
    farmer: dict = Depends(get_current_farmer)
):
    """Returns the farmer's last N yield predictions."""
    records = get_yield_history(farmer["id"], limit=limit)
    return {"success": True, "data": records}

@router.get("/yield-history/{prediction_id}")
def get_single_yield(
    prediction_id: str,
    farmer: dict = Depends(get_current_farmer)
):
    result = get_yield_by_id(farmer["id"], prediction_id)
    if not result:
        raise HTTPException(status_code=404, detail="Prediction not found.")
    return {"success": True, "data": result}

@router.get("/yield-options")
def yield_options():
    """Valid crops, seasons, and states for dropdowns. Public endpoint."""
    try:
        return get_yield_options()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))