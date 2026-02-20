import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Import konfigurasi database dan tabel models
from database import engine, Base
import models

# Import routers dari folder routers
# Pastikan Anda sudah membuat file-file ini di folder /routers
from routers import mangrove, rice, portfolio
from routers.mangrove import router as mangrove_router
from routers.rice import router as rice_router
from routers.portfolio import router as portfolio_router
from routers.login import router as login_router

# 1. Memuat variabel lingkungan dari file .env
load_dotenv()

# 2. Inisialisasi Aplikasi FastAPI
ENV = os.getenv("ENV", "dev")
app = FastAPI(
    title="Saifudin Nasir Professional Portfolio API",
    description="Backend terintegrasi untuk riset Machine Learning, IoT, dan Manajemen Proyek",
    version="2.0.0",
    docs_url="/docs" if ENV == "dev" else None, # Dokumentasi Swagger hanya muncul di mode dev
    redoc_url=None
)

# 3. Inisialisasi Database (Otomatis membuat tabel di pgAdmin/PostgreSQL)
# Baris ini akan memindai models.py dan membuat tabel jika belum ada
Base.metadata.create_all(bind=engine)

# 4. Konfigurasi CORS (Cross-Origin Resource Sharing)
# Mengizinkan Frontend (React/Next.js) mengakses API ini
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:3000")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN, "http://127.0.0.1:3000"],
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

# 6. Endpoint Dasar untuk Pengecekan Sistem
@app.get("/", tags=["System Check"])
def root():
    return {
        "status": "online",
        "message": "Portfolio API Hub is running successfully",
        "environment": ENV
    }

# 7. Eksekusi Server
if __name__ == "__main__":
    # Gunakan reload=True saat pengembangan agar server restart otomatis saat kode diubah
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=(ENV == "dev"))