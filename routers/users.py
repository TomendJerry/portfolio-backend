import os
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional
from passlib.context import CryptContext
from jose import jwt, JWTError

from database import SessionLocal
import models

router = APIRouter(
    prefix="/api/v1/users",
    tags=["User Management (CRUD)"]
)

# ==========================================
# KONFIGURASI KEAMANAN (Bcrypt & JWT)
# ==========================================
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

SECRET_KEY = os.getenv("SECRET_KEY", "kunci_rahasia_cadangan_jika_env_kosong")
ALGORITHM = "HS256"

# Memberitahu FastAPI/Swagger UI di mana URL untuk mengambil token
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")


# ==========================================
# DEPENDENCIES (Database & Keamanan)
# ==========================================
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    """
    Fungsi 'Satpam' untuk memverifikasi JWT Token di setiap request.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Akses ditolak: Token tidak valid atau sudah kedaluwarsa",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        # Dekode (buka) token
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    # Cek apakah user dengan token ini benar-benar ada di database
    user = db.query(models.User).filter(models.User.username == username).first()
    if user is None:
        raise credentials_exception
    
    return user

def check_super_admin(current_user: models.User = Depends(get_current_user)):
    """
    Hanya membolehkan akses jika role user adalah 'super_admin'.
    """
    if current_user.role != "super_admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Akses Ditolak: Anda tidak memiliki hak akses Super Admin"
        )
    return current_user
# ==========================================
# SCHEMA (Bentuk Data Input & Output)
# ==========================================
class UserBase(BaseModel):
    username: str
    full_name: str
    role: str = "admin"

class UserCreate(UserBase):
    password: str  

class UserUpdate(BaseModel):
    full_name: Optional[str] = None
    role: Optional[str] = None
    password: Optional[str] = None 

class UserResponse(UserBase):
    id: int
    class Config:
        from_attributes = True  


# ==========================================
# ENDPOINTS (Jalur API yang Sudah Digembok)
# ==========================================

# 1. READ ALL - Wajib menyertakan Token (Depends(get_current_user))
@router.get("/", response_model=List[UserResponse])
def get_all_users(
    db: Session = Depends(get_db), 
    current_user: models.User = Depends(check_super_admin)
):
    users = db.query(models.User).all()
    return users

# 2. CREATE - Wajib menyertakan Token
@router.post("/", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def create_user(
    user: UserCreate, 
    db: Session = Depends(get_db),
    current_user: models.User = Depends(check_super_admin) # GEMBOK DIPASANG DI SINI
):
    existing_user = db.query(models.User).filter(models.User.username == user.username).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Username sudah terdaftar!")
    
    hashed_password = pwd_context.hash(user.password)
    
    new_user = models.User(
        username=user.username,
        full_name=user.full_name,
        role=user.role,
        password_hash=hashed_password
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    return new_user

# 3. UPDATE - Wajib menyertakan Token
@router.put("/{user_id}", response_model=UserResponse)
def update_user(
    user_id: int, 
    user_update: UserUpdate, 
    db: Session = Depends(get_db),
    current_user: models.User = Depends(check_super_admin) # GEMBOK DIPASANG DI SINI
):
    db_user = db.query(models.User).filter(models.User.id == user_id).first()
    if not db_user:
        raise HTTPException(status_code=404, detail="User tidak ditemukan")
    
    if user_update.full_name is not None:
        db_user.full_name = user_update.full_name
    if user_update.role is not None:
        db_user.role = user_update.role
    if user_update.password is not None:
        db_user.password_hash = pwd_context.hash(user_update.password)
        
    db.commit()
    db.refresh(db_user)
    return db_user

# 4. DELETE - Wajib menyertakan Token
@router.delete("/{user_id}")
def delete_user(
    user_id: int, 
    db: Session = Depends(get_db),
    current_user: models.User = Depends(check_super_admin) # GEMBOK DIPASANG DI SINI
):
    # Fitur Keamanan Tambahan: Mencegah admin menghapus dirinya sendiri secara tidak sengaja
    if current_user.id == user_id:
        raise HTTPException(status_code=400, detail="Anda tidak bisa menghapus akun Anda sendiri yang sedang aktif.")

    db_user = db.query(models.User).filter(models.User.id == user_id).first()
    if not db_user:
        raise HTTPException(status_code=404, detail="User tidak ditemukan")
    
    db.delete(db_user)
    db.commit()
    return {"status": "success", "message": f"User '{db_user.username}' berhasil dihapus."}