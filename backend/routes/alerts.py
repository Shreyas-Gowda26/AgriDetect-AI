"""
routes/alerts.py
----------------
SMS alert endpoint — protected route.
Farmer triggers irrigation alert manually.
Weather is auto-fetched, SMS sent to farmer's registered phone.
"""

from fastapi import APIRouter, HTTPException, Depends
from routes.auth     import get_current_farmer
from weather_service import get_weather_by_city
from sms_service     import send_irrigation_alert

router = APIRouter()


@router.post("/send-irrigation-alert")
async def irrigation_alert(
    farmer: dict = Depends(get_current_farmer)
):
    """
    Send irrigation alert SMS to the farmer's registered phone number.
    Weather is auto-fetched based on farmer's saved city.
    No input needed — just hit the endpoint!
    """
    # Step 1: Fetch weather using farmer's saved city
    try:
        weather = await get_weather_by_city(farmer["city"])
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    # Step 2: Send SMS
    try:
        result = send_irrigation_alert(
            farmer_name=farmer["full_name"],
            farmer_phone=farmer["phone"],
            city=weather["city"],
            temperature=weather["temperature"],
            humidity=weather["humidity"],
            rainfall=weather["rainfall"],
            weather_desc=weather["weather_desc"],
        )
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "success": True,
        "message": f"Irrigation alert sent to {farmer['phone']}",
        "data": {
            "sent_to":     result["sent_to"],
            "status":      result["status"],
            "message_sid": result["message_sid"],
            "weather": {
                "city":        weather["city"],
                "temperature": weather["temperature"],
                "humidity":    weather["humidity"],
                "rainfall":    weather["rainfall"],
                "condition":   weather["weather_desc"],
            },
            "sms_preview": result["preview"],
        }
    }