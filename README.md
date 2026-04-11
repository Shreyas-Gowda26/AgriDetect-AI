# 🌾 AgriDetect-AI

> AI-powered precision agriculture platform for Indian farmers — crop recommendation, yield prediction, disease detection, and smart irrigation alerts.

---

## 🚀 Features

### 🌱 Crop Recommendation
- Recommends the best crop based on soil nutrients (N, P, K, pH) and real-time weather
- Soil profile saved once — farmer only needs to enter their city next time
- **99.55% accuracy** using Random Forest Classifier trained on 2,200 samples across 22 crops

### 📈 Yield Prediction
- Predicts crop yield in tons/hectare
- Inputs: crop, season, area, fertilizer, pesticide
- Annual rainfall auto-fetched from weather API
- **94.25% accuracy (R² = 0.9425)** using XGBoost Regressor trained on 19,689 rows across 55 crops and 30 Indian states

### 🔬 Disease Detection
- Upload a leaf image → get disease name, cause, pesticide recommendation, dosage, and prevention tips
- **38 disease classes across 14 crops** using MobileNetV2 pretrained on PlantVillage dataset
- Returns top 3 predictions with confidence scores
- Leaf images stored securely in Supabase Storage

### 💧 Smart Irrigation Alerts
- One-tap SMS alert to farmer's phone
- Weather-based irrigation advice (urgent / moderate / none)
- Powered by Twilio + OpenWeatherMap

### 🌍 Smart Weather Integration
- Real-time weather auto-fetched using OpenWeatherMap API
- Farmer just enters city name — temperature, humidity, rainfall handled automatically
- Annual rainfall estimated from city-level historical data

### 📋 Prediction History
- Every prediction saved per farmer in Supabase
- View full history or fetch a specific prediction by ID
- Supports yield, crop recommendation, and disease detection history

### 🔒 Secure Auth
- JWT-based authentication
- Phone number as primary identifier (suited for Indian farmers)
- bcrypt password hashing
- All routes protected — farmers only see their own data

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI (Python) |
| Database | Supabase (PostgreSQL) |
| Auth | JWT (python-jose + bcrypt) |
| ML — Tabular | scikit-learn, XGBoost |
| ML — Vision | PyTorch, MobileNetV2 |
| Weather | OpenWeatherMap API |
| SMS Alerts | Twilio |
| Image Storage | Supabase Storage |

---

## 🤖 ML Models

| Model | Algorithm | Dataset | Metric |
|---|---|---|---|
| Crop Recommendation | Random Forest Classifier | 2,200 rows, 22 crops | **99.55% accuracy** |
| Yield Prediction | XGBoost Regressor | 19,689 rows, 55 crops, 30 states | **R² = 0.9425** |
| Disease Detection | MobileNetV2 (Transfer Learning) | PlantVillage, 87k+ images | **38 disease classes** |

### Feature Importance (Yield Model)
```
Crop type       → 57.36%
Season          → 13.08%
State           →  8.98%
Area            →  5.57%
Fertilizer      →  5.42%
Rainfall        →  4.38%
Pesticide       →  3.82%
Year            →  1.40%
```

---

## 📁 Project Structure

```
AgriDetect-AI/
└── backend/
    ├── main.py                      # FastAPI app entry point
    ├── ml_service.py                # All 3 ML models (crop rec, yield, disease)
    ├── weather_service.py           # OpenWeatherMap integration
    ├── supabase_service.py          # All DB operations
    ├── sms_service.py               # Twilio SMS alerts
    ├── crop_yield_model.py          # Yield model training script
    ├── crop_recommendation_model.py # Crop rec model training script
    ├── requirements.txt
    └── routes/
        ├── auth.py                  # Register, login, me
        ├── soil.py                  # Soil profile (POST, GET, PATCH)
        ├── crop_rec.py              # Crop recommendation
        ├── yield_route.py           # Yield prediction
        ├── disease.py               # Disease detection
        └── alerts.py                # Irrigation SMS alerts
```

---

## ⚡ API Endpoints

```
AUTH
  POST  /api/auth/register
  POST  /api/auth/login
  GET   /api/auth/me

SOIL PROFILE
  POST  /api/soil-profile          ← save once after soil testing
  GET   /api/soil-profile          ← fetch saved values
  PATCH /api/soil-profile          ← update specific values (e.g. just pH)

CROP RECOMMENDATION
  POST  /api/recommend-crop        ← just city needed!
  GET   /api/crop-rec-history
  GET   /api/crop-rec-history/{id}
  GET   /api/crop-rec-options

YIELD PREDICTION
  POST  /api/predict-yield
  GET   /api/yield-history
  GET   /api/yield-history/{id}
  GET   /api/yield-options

DISEASE DETECTION
  POST  /api/detect-disease
  GET   /api/disease-history
  GET   /api/disease-history/{id}
  GET   /api/supported-crops

ALERTS
  POST  /api/send-irrigation-alert ← one tap, SMS sent instantly
```

Full interactive docs available at `/docs` when server is running.

---

## 🏃 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/Shreyas-Gowda26/AgriDetect-AI.git
cd AgriDetect-AI/backend
```

### 2. Create virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
Create a `.env` file inside `backend/`:
```
SUPABASE_URL=your_supabase_url
SUPABASE_SERVICE_KEY=your_service_role_key
OPENWEATHER_API_KEY=your_openweather_key
TWILIO_ACCOUNT_SID=your_twilio_sid
TWILIO_AUTH_TOKEN=your_twilio_token
TWILIO_PHONE_NUMBER=your_twilio_number
SECRET_KEY=your_jwt_secret_key
```

### 5. Train ML models
```bash
python crop_yield_model.py
python crop_recommendation_model.py
```
This generates `.pkl` files in `ml_models/`.
The disease detection model (~50MB) downloads automatically on first run.

### 6. Run the server
```bash
python -m uvicorn main:app --reload
```

### 7. Open API docs
```
http://localhost:8000/docs
```

---

## 🗄️ Database Schema

Run `migration.sql` in your Supabase SQL Editor:

| Table | Purpose |
|---|---|
| `farmers` | Auth — stores farmer profile + hashed password |
| `soil_profiles` | One soil profile per farmer |
| `yield_predictions` | Full history of yield predictions |
| `crop_recommendations` | Full history of crop recommendations |
| `disease_detections` | Full history + leaf image URLs |

---

## 🌾 How It Works

```
Farmer registers → saves soil profile once
        ↓
Opens app → types city name
        ↓
Backend fetches live weather automatically
        ↓
ML model runs → recommends best crop
        ↓
Result saved to Supabase
        ↓
Farmer can view full history anytime
        ↓
One tap → SMS irrigation alert on phone
```

---

## 👥 Team

Built with a team of 4 members 

**Team Predators**

- Shreyas G
- Shreyas J S
- Roshan
- Darshan Kudrigi

---

## 📄 License

MIT License