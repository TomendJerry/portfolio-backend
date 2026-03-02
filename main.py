import os
import uvicorn
import models

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from routers.users import router as users_router
from routers import mangrove, rice, portfolio, rating
from routers.mangrove import router as mangrove_router
from routers.rice import router as rice_router
from routers.portfolio import router as portfolio_router
from routers.login import router as login_router
from routers.resume import router as resume_router
from database import engine, Base
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi.staticfiles import StaticFiles

# 1. Memuat variabel lingkungan dari file .env
load_dotenv()

# 2. Inisialisasi Aplikasi FastAPI & Rate Limiter (Anti DDoS/Spam)
limiter = Limiter(key_func=get_remote_address)
ENV = os.getenv("ENV", "dev")

app = FastAPI(
    title="Saifudin Nasir Professional Portfolio API",
    description="Backend terintegrasi dengan Keamanan Standar Industri",
    version="2.1.0",
    docs_url="/docs" if ENV == "dev" else None,
    redoc_url=None
)

# Daftarkan penangkap error jika ada user yang melebihi batas request
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# 3. Inisialisasi Database (Otomatis membuat tabel di pgAdmin/PostgreSQL)
# Baris ini akan memindai models.py dan membuat tabel jika belum ada
Base.metadata.create_all(bind=engine)

# 4. Konfigurasi CORS (Cross-Origin Resource Sharing)
# Mengizinkan Frontend (React/Next.js) mengakses API ini
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        FRONTEND_URL, 
        "http://localhost:3000",   # Tambahkan ini untuk keamanan lokal
        "http://127.0.0.1:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 5. Pendaftaran Router (Modularisasi)
# Setiap modul memiliki file sendiri agar kode tidak menumpuk di main.py
app.include_router(mangrove.router)   # Modul Skripsi & Carbon
app.include_router(rice.router)       # Modul Rice Supply Chain
app.include_router(portfolio.router)  # Modul Manajemen Proyek & Profil
app.include_router(login_router)  # Mendaftarkan router autentikasi login
app.include_router(users_router)
app.include_router(resume_router)
app.include_router(rating.router)

# 6. Endpoint Dasar untuk Pengecekan Sistem (Dibatasi 5 kali per menit per IP)
@app.get("/", tags=["System Check"])
@limiter.limit("5/minute")
def root(request: Request):
    return {
        "status": "online",
        "message": "Portfolio API Hub is running securely",
        "environment": ENV
    }

# 7. Eksekusi Server
if __name__ == "__main__":
    # Gunakan reload=True saat pengembangan agar server restart otomatis saat kode diubah
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=(ENV == "dev"))