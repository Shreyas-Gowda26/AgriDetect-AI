"""
weather_service.py
------------------
Fetches real-time weather data from OpenWeatherMap API using city name.
Used to auto-fill weather inputs for crop recommendation and yield prediction
so farmers don't have to manually enter weather data.

Add to .env:
    OPENWEATHER_API_KEY=your_api_key_here

Get a free API key at: https://openweathermap.org/api
"""

import os
import httpx
from dotenv import load_dotenv

load_dotenv()

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
BASE_URL = "https://api.openweathermap.org/data/2.5"


async def get_weather_by_city(city: str) -> dict:
    """
    Fetch current weather data for a given city name.

    Args:
        city: City or village name e.g. "Mysuru", "Pune", "Warangal"

    Returns:
        {
            "city": str,
            "state": str,           ← mapped Indian state name
            "country": str,
            "temperature": float,   ← Celsius
            "humidity": float,      ← percentage
            "rainfall": float,      ← mm (last 1 hour, 0 if no rain)
            "weather_desc": str,    ← e.g. "light rain", "clear sky"
            "annual_rainfall_estimate": float  ← rough annual estimate in mm
        }

    Raises:
        ValueError: if city not found
        RuntimeError: if API key missing or request fails
    """
    if not OPENWEATHER_API_KEY:
        raise RuntimeError(
            "OPENWEATHER_API_KEY not set in .env file. "
            "Get a free key at https://openweathermap.org/api"
        )

    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(
            f"{BASE_URL}/weather",
            params={
                "q":     city,
                "appid": OPENWEATHER_API_KEY,
                "units": "metric",   # Celsius
            }
        )

    if response.status_code == 404:
        raise ValueError(
            f"City '{city}' not found. Please check the spelling or try a nearby larger city."
        )
    if response.status_code == 401:
        raise RuntimeError("Invalid OpenWeatherMap API key. Check your .env file.")
    if response.status_code != 200:
        raise RuntimeError(f"Weather API error: {response.status_code} — {response.text}")

    data = response.json()

    temperature  = round(data["main"]["temp"], 1)
    humidity     = round(data["main"]["humidity"], 1)
    weather_desc = data["weather"][0]["description"]

    # Rainfall: OpenWeatherMap returns rain.1h if it's raining, else no "rain" key
    rainfall_1h = 0.0
    if "rain" in data:
        rainfall_1h = data["rain"].get("1h", 0.0)

    # Rough annual rainfall estimate based on current rainfall rate
    # This is a simple heuristic — for production use a historical API
    annual_rainfall_estimate = _estimate_annual_rainfall(
        city=city,
        current_rainfall_mm=rainfall_1h,
        humidity=humidity
    )

    # Try to map to Indian state name for yield prediction
    state = _map_city_to_state(city)

    return {
        "city":                     data["name"],
        "state":                    state,
        "country":                  data["sys"]["country"],
        "temperature":              temperature,
        "humidity":                 humidity,
        "rainfall":                 rainfall_1h,
        "weather_desc":             weather_desc,
        "annual_rainfall_estimate": annual_rainfall_estimate,
    }


def _estimate_annual_rainfall(city: str, current_rainfall_mm: float, humidity: float) -> float:
    """
    Estimate annual rainfall in mm using known averages for major Indian cities.
    Falls back to a humidity-based estimate for unknown cities.

    This is used as input for the yield prediction model which expects annual rainfall.
    """
    # Known annual rainfall averages (mm) for major Indian cities/regions
    CITY_RAINFALL = {
        # Karnataka
        "mysuru": 786, "mysore": 786, "bangalore": 970, "bengaluru": 970,
        "mangalore": 3500, "hubli": 840, "dharwad": 840, "belgaum": 1100,
        # Maharashtra
        "mumbai": 2167, "pune": 722, "nagpur": 1034, "nashik": 680,
        "aurangabad": 726, "solapur": 547,
        # Tamil Nadu
        "chennai": 1400, "coimbatore": 686, "madurai": 850, "trichy": 843,
        # Andhra Pradesh / Telangana
        "hyderabad": 812, "warangal": 1016, "vijayawada": 1067, "visakhapatnam": 1035,
        # Kerala
        "kochi": 3128, "thiruvananthapuram": 1757, "kozhikode": 2948,
        # Uttar Pradesh
        "lucknow": 897, "kanpur": 770, "varanasi": 1102, "agra": 688,
        # Punjab / Haryana
        "amritsar": 682, "ludhiana": 733, "chandigarh": 1100,
        # Rajasthan
        "jaipur": 650, "jodhpur": 362, "udaipur": 612,
        # Gujarat
        "ahmedabad": 782, "surat": 1143, "vadodara": 934,
        # West Bengal
        "kolkata": 1582, "siliguri": 2800,
        # Madhya Pradesh
        "bhopal": 1146, "indore": 961, "jabalpur": 1390,
        # Odisha
        "bhubaneswar": 1500, "cuttack": 1400,
        # Bihar
        "patna": 1200,
        # Jharkhand
        "ranchi": 1430,
        # Assam
        "guwahati": 1600,
        # Delhi
        "delhi": 714, "new delhi": 714,
    }

    city_lower = city.lower().strip()
    if city_lower in CITY_RAINFALL:
        return float(CITY_RAINFALL[city_lower])

    # Humidity-based fallback estimate
    # High humidity regions tend to have higher rainfall
    if humidity >= 80:
        return 1500.0
    elif humidity >= 65:
        return 1000.0
    elif humidity >= 50:
        return 700.0
    else:
        return 400.0


def _map_city_to_state(city: str) -> str:
    """
    Map city name to Indian state name as used in the yield prediction model.
    Returns empty string if not found — frontend can let user confirm/override.
    """
    CITY_TO_STATE = {
        # Karnataka
        "mysuru": "Karnataka", "mysore": "Karnataka", "bangalore": "Karnataka",
        "bengaluru": "Karnataka", "mangalore": "Karnataka", "hubli": "Karnataka",
        "dharwad": "Karnataka", "belgaum": "Karnataka", "davangere": "Karnataka",
        "shimoga": "Karnataka", "tumkur": "Karnataka",
        # Maharashtra
        "mumbai": "Maharashtra", "pune": "Maharashtra", "nagpur": "Maharashtra",
        "nashik": "Maharashtra", "aurangabad": "Maharashtra", "solapur": "Maharashtra",
        "kolhapur": "Maharashtra", "amravati": "Maharashtra",
        # Tamil Nadu
        "chennai": "Tamil Nadu", "coimbatore": "Tamil Nadu", "madurai": "Tamil Nadu",
        "trichy": "Tamil Nadu", "salem": "Tamil Nadu", "tirunelveli": "Tamil Nadu",
        # Andhra Pradesh
        "vijayawada": "Andhra Pradesh", "visakhapatnam": "Andhra Pradesh",
        "guntur": "Andhra Pradesh", "nellore": "Andhra Pradesh",
        # Telangana
        "hyderabad": "Telangana", "warangal": "Telangana", "nizamabad": "Telangana",
        # Kerala
        "kochi": "Kerala", "thiruvananthapuram": "Kerala", "kozhikode": "Kerala",
        "thrissur": "Kerala", "kollam": "Kerala",
        # Uttar Pradesh
        "lucknow": "Uttar Pradesh", "kanpur": "Uttar Pradesh", "varanasi": "Uttar Pradesh",
        "agra": "Uttar Pradesh", "allahabad": "Uttar Pradesh", "meerut": "Uttar Pradesh",
        # Punjab
        "amritsar": "Punjab", "ludhiana": "Punjab", "jalandhar": "Punjab",
        # Haryana
        "chandigarh": "Haryana", "faridabad": "Haryana", "gurgaon": "Haryana",
        # Rajasthan
        "jaipur": "Rajasthan", "jodhpur": "Rajasthan", "udaipur": "Rajasthan",
        "kota": "Rajasthan", "ajmer": "Rajasthan",
        # Gujarat
        "ahmedabad": "Gujarat", "surat": "Gujarat", "vadodara": "Gujarat",
        "rajkot": "Gujarat", "gandhinagar": "Gujarat",
        # West Bengal
        "kolkata": "West Bengal", "siliguri": "West Bengal", "asansol": "West Bengal",
        # Madhya Pradesh
        "bhopal": "Madhya Pradesh", "indore": "Madhya Pradesh", "jabalpur": "Madhya Pradesh",
        "gwalior": "Madhya Pradesh",
        # Odisha
        "bhubaneswar": "Odisha", "cuttack": "Odisha", "rourkela": "Odisha",
        # Bihar
        "patna": "Bihar", "gaya": "Bihar", "muzaffarpur": "Bihar",
        # Jharkhand
        "ranchi": "Jharkhand", "jamshedpur": "Jharkhand", "dhanbad": "Jharkhand",
        # Assam
        "guwahati": "Assam", "dibrugarh": "Assam",
        # Delhi
        "delhi": "Uttar Pradesh", "new delhi": "Uttar Pradesh",
        # Himachal Pradesh
        "shimla": "Himachal Pradesh", "manali": "Himachal Pradesh",
        # Uttarakhand
        "dehradun": "Uttarakhand", "haridwar": "Uttarakhand",
        # Chhattisgarh
        "raipur": "Chhattisgarh", "bilaspur": "Chhattisgarh",
    }

    return CITY_TO_STATE.get(city.lower().strip(), "")