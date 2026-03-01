import os
from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from jose import jwt
from passlib.context import CryptContext
from database import SessionLocal
import models

router = APIRouter(
    prefix="/api/v1/auth",
    tags=["Authentication"]
)

# --- Konfigurasi Keamanan ---
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

SECRET_KEY = os.getenv("SECRET_KEY", "kunci_rahasia_cadangan_jika_env_kosong")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 120 

# --- Helper Functions ---
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# --- Endpoint API ---
# PERUBAHAN UTAMA: Menggunakan OAuth2PasswordRequestForm sebagai ganti LoginRequest
@router.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    """
    Endpoint untuk autentikasi user ke sistem Dashboard Admin (Bcrypt + JWT Secured).
    Kompatibel dengan sistem Form-Data bawaan Swagger UI.
    """
    # 1. Cari user berdasarkan username (menggunakan form_data.username)
    user = db.query(models.User).filter(models.User.username == form_data.username).first()
    
    if not user:
        raise HTTPException(status_code=401, detail="Kredensial tidak valid")

    # 2. Verifikasi Password menggunakan Bcrypt (menggunakan form_data.password)
    try:
        is_valid = pwd_context.verify(form_data.password, user.password_hash)
    except Exception:
        is_valid = False

    if not is_valid:
        raise HTTPException(status_code=401, detail="Kredensial tidak valid")

    # 3. Buat JWT Token jika sukses
    access_token = create_access_token(data={"sub": user.username})

    # PENTING: Response harus mengembalikan access_token dan token_type agar dibaca oleh Swagger
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "status": "success",
        "message": "Login Berhasil",
        "user": {
            "username": user.username, 
            "full_name": user.full_name,
            "role": user.role
        }
    }