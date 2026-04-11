"""
routes/auth.py
--------------
Authentication routes for AgriPredict.
  POST /api/auth/register  → create account, return JWT
  POST /api/auth/login     → verify credentials, return JWT
  GET  /api/auth/me        → return logged-in farmer profile
"""

from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from jose import JWTError, jwt
from passlib.context import CryptContext
import os

from supabase_service import create_farmer, get_farmer_by_phone, get_farmer_by_id

router   = APIRouter()
security = HTTPBearer()

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
SECRET_KEY      = os.getenv("SECRET_KEY", "change-this-in-production")
ALGORITHM       = "HS256"
TOKEN_EXPIRE_DAYS = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)

def create_jwt(farmer_id: str) -> str:
    expire  = datetime.utcnow() + timedelta(days=TOKEN_EXPIRE_DAYS)
    payload = {"sub": farmer_id, "exp": expire}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def decode_jwt(token: str) -> str:
    """Returns farmer_id from token or raises HTTPException."""
    try:
        payload   = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        farmer_id = payload.get("sub")
        if not farmer_id:
            raise HTTPException(status_code=401, detail="Invalid token.")
        return farmer_id
    except JWTError:
        raise HTTPException(status_code=401, detail="Token expired or invalid.")


# ─────────────────────────────────────────────
# Dependency — use this in all protected routes
# ─────────────────────────────────────────────
async def get_current_farmer(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> dict:
    """
    FastAPI dependency. Validates JWT and returns farmer dict.

    Usage in any route:
        @router.post("/some-route")
        async def route(farmer: dict = Depends(get_current_farmer)):
            farmer_id = farmer["id"]
    """
    farmer_id = decode_jwt(credentials.credentials)
    farmer    = get_farmer_by_id(farmer_id)
    if not farmer:
        raise HTTPException(status_code=401, detail="Farmer account not found.")
    return farmer


# ─────────────────────────────────────────────
# Request / Response schemas
# ─────────────────────────────────────────────
class RegisterRequest(BaseModel):
    full_name: str   = Field(..., example="Ravi Kumar")
    phone:     str   = Field(..., example="9876543210")
    password:  str   = Field(..., min_length=6, example="securepass123")
    city:      str   = Field(..., example="Mysuru")
    state:     str   = Field(..., example="Karnataka")

class LoginRequest(BaseModel):
    phone:    str = Field(..., example="9876543210")
    password: str = Field(..., example="securepass123")

class AuthResponse(BaseModel):
    token:     str
    farmer_id: str
    full_name: str
    phone:     str
    city:      str
    state:     str


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@router.post("/auth/register", response_model=AuthResponse)
def register(req: RegisterRequest):
    """
    Register a new farmer account.
    Returns a JWT token valid for 30 days.
    """
    # Check if phone already registered
    existing = get_farmer_by_phone(req.phone)
    if existing:
        raise HTTPException(
            status_code=409,
            detail="This phone number is already registered. Please login instead."
        )

    # Hash password and save farmer
    hashed  = hash_password(req.password)
    farmer  = create_farmer(
        full_name=req.full_name,
        phone=req.phone,
        password_hash=hashed,
        city=req.city,
        state=req.state,
    )

    token = create_jwt(farmer["id"])

    return AuthResponse(
        token=token,
        farmer_id=farmer["id"],
        full_name=farmer["full_name"],
        phone=farmer["phone"],
        city=farmer["city"],
        state=farmer["state"],
    )


@router.post("/auth/login", response_model=AuthResponse)
def login(req: LoginRequest):
    """
    Login with phone + password.
    Returns a JWT token valid for 30 days.
    """
    farmer = get_farmer_by_phone(req.phone)

    # Same error for both "not found" and "wrong password" — prevents user enumeration
    if not farmer or not verify_password(req.password, farmer["password_hash"]):
        raise HTTPException(
            status_code=401,
            detail="Incorrect phone number or password."
        )

    token = create_jwt(farmer["id"])

    return AuthResponse(
        token=token,
        farmer_id=farmer["id"],
        full_name=farmer["full_name"],
        phone=farmer["phone"],
        city=farmer["city"],
        state=farmer["state"],
    )


@router.get("/auth/me")
def me(farmer: dict = Depends(get_current_farmer)):
    """
    Returns the currently logged-in farmer's profile.
    Requires Authorization: Bearer <token> header.
    """
    return {
        "farmer_id": farmer["id"],
        "full_name": farmer["full_name"],
        "phone":     farmer["phone"],
        "city":      farmer["city"],
        "state":     farmer["state"],
        "member_since": farmer["created_at"],
    }