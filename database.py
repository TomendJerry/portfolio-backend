import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv

# Memuat variabel dari file .env
load_dotenv()

# Mengambil konfigurasi dari Environment Variables
ENV = os.getenv("ENV", "dev")
DATABASE_URL = os.getenv("DATABASE_URL")

# Validasi DATABASE_URL
if not DATABASE_URL:
    if ENV == "dev":
        # Fallback default untuk pengembangan lokal jika .env tidak terbaca
        DATABASE_URL = "postgresql+psycopg2://postgres:nasir8@localhost:5433/rice_db"
    else:
        # Keamanan ketat untuk mode Production
        raise RuntimeError("DATABASE_URL environment variable must be set in production")

# Perbaikan format URL jika menggunakan platform seperti Heroku/Render
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+psycopg2://", 1)

# Inisialisasi Engine SQLAlchemy
engine = create_engine(
    DATABASE_URL,
    # Menampilkan log kueri SQL di terminal hanya pada mode dev
    echo=(ENV == "dev"),
    pool_pre_ping=True  # Memastikan koneksi tetap hidup
)

# Inisialisasi Sesi Database
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class untuk mendefinisikan model di models.py
Base = declarative_base()