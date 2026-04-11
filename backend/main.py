"""
main.py
-------
AgriPredict FastAPI application entry point.

Run with:
    /Users/shreyasg/Desktop/agri/.venv/bin/python -m uvicorn main:app --reload
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

from routes.auth         import router as auth_router
from routes.soil         import router as soil_router
from routes.yield_route  import router as yield_router
from routes.crop_rec     import router as crop_rec_router
from routes.disease      import router as disease_router
from routes.alerts       import router as alerts_router
app = FastAPI(
    title="AgriPredict API",
    description="AI-powered crop yield prediction, crop recommendation, and disease detection for Indian farmers.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router,     prefix="/api", tags=["Auth"])
app.include_router(soil_router,     prefix="/api", tags=["Soil Profile"])
app.include_router(yield_router,    prefix="/api", tags=["Yield Prediction"])
app.include_router(crop_rec_router, prefix="/api", tags=["Crop Recommendation"])
app.include_router(disease_router,  prefix="/api", tags=["Disease Detection"])
app.include_router(alerts_router, prefix="/api", tags=["Alerts"])

@app.get("/", tags=["Health"])
def root():
    return {
        "status":  "running",
        "message": "AgriPredict API is live 🌾",
        "docs":    "/docs",
    }

@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok"}

