"""
sms_service.py
--------------
Twilio SMS service for AgriPredict.
Sends irrigation advice + weather forecast to farmers.

Add to .env:
    TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    TWILIO_AUTH_TOKEN=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    TWILIO_PHONE_NUMBER=+1xxxxxxxxxx
"""

import os
from twilio.rest import Client
from dotenv import load_dotenv

load_dotenv()

TWILIO_ACCOUNT_SID  = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN   = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")


def _get_client() -> Client:
    if not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN:
        raise RuntimeError(
            "TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN must be set in your .env file."
        )
    return Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)


def _build_irrigation_message(
    farmer_name: str,
    city: str,
    temperature: float,
    humidity: float,
    rainfall: float,
    weather_desc: str,
) -> str:
    """
    Build short irrigation advice message based on current weather.
    Kept short (1-2 segments) for Twilio trial account compatibility.
    """
    if rainfall > 10:
        irrigation_emoji  = "✅"
        irrigation_advice = "No irrigation needed. Rainfall is sufficient."
    elif humidity >= 70 and rainfall > 2:
        irrigation_emoji  = "💧"
        irrigation_advice = "Light irrigation recommended. Water fields in the evening."
    elif temperature > 30 and humidity < 50:
        irrigation_emoji  = "🚨"
        irrigation_advice = "Urgent: Heavy irrigation needed! Water fields immediately."
    else:
        irrigation_emoji  = "💧"
        irrigation_advice = "Moderate irrigation recommended. Best time: early morning."

    message = f"""AgriPredict Alert - {farmer_name}

Location: {city}
Temp: {temperature}C | Humidity: {humidity}%
Condition: {weather_desc.capitalize()}

{irrigation_emoji} {irrigation_advice}

- AgriPredict Team"""

    return message


def send_irrigation_alert(
    farmer_name: str,
    farmer_phone: str,
    city: str,
    temperature: float,
    humidity: float,
    rainfall: float,
    weather_desc: str,
) -> dict:
    """
    Send irrigation alert SMS to farmer.

    Args:
        farmer_name  : Farmer's full name
        farmer_phone : Farmer's phone number e.g. "9876543210"
        city         : Farmer's city
        temperature  : Current temperature in Celsius
        humidity     : Current humidity percentage
        rainfall     : Current rainfall in mm
        weather_desc : Weather description e.g. "light rain"

    Returns:
        { "success": bool, "message_sid": str, "status": str }
    """
    # Format to E.164 (+91XXXXXXXXXX for India)
    phone = farmer_phone.strip()
    if not phone.startswith("+"):
        phone = f"+91{phone}"

    print(f"📱 Sending SMS to: {phone}")

    message_body = _build_irrigation_message(
        farmer_name=farmer_name,
        city=city,
        temperature=temperature,
        humidity=humidity,
        rainfall=rainfall,
        weather_desc=weather_desc,
    )

    try:
        client  = _get_client()
        message = client.messages.create(
            body=message_body,
            from_=TWILIO_PHONE_NUMBER,
            to=phone,
        )
        print(f"✅ SMS sent! SID: {message.sid} | Status: {message.status}")
        return {
            "success":     True,
            "message_sid": message.sid,
            "status":      message.status,
            "sent_to":     phone,
            "preview":     message_body,
        }
    except Exception as e:
        raise RuntimeError(f"Failed to send SMS: {str(e)}")