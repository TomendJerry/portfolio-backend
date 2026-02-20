import hashlib
from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel
from sqlalchemy.orm import Session
from database import SessionLocal
import models

router = APIRouter(
    prefix="/api/v1/auth",
    tags=["Authentication"]
)

# --- Schema Data (Pydantic) ---
class LoginRequest(BaseModel):
    username: str
    password: str

# --- Dependency untuk mendapatkan Session Database ---
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- Endpoint API ---
@router.post("/login")
def login(data: LoginRequest, db: Session = Depends(get_db)):
    """
    Endpoint untuk autentikasi user ke sistem Dashboard Admin.
    """
    # 1. Cari user berdasarkan username
    user = db.query(models.User).filter(models.User.username == data.username).first()
    
    if not user:
        raise HTTPException(status_code=401, detail="Username_Salah")

    # 2. Verifikasi Password (SHA-256)
    hashed_input = hashlib.sha256(data.password.encode()).hexdigest()
    # DEBUG: Cek di terminal apakah keduanya sama
    print(f"DEBUG: Hash dari Input: {hashed_input}")
    print(f"DEBUG: Hash di Database: {user.password_hash}")
    
    if hashed_input != user.password_hash:
        raise HTTPException(status_code=401, detail="Password_Salah")

    return {
        "status": "success",
        "message": "Login_Berhasil",
        "user": {
            "username": user.username, 
            "full_name": user.full_name
        }
    }