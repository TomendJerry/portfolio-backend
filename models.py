from sqlalchemy import Column, Integer, String, Float, Boolean, Text, DateTime
from database import Base
import datetime
import uuid

# ==========================================
# 1. CORE PORTFOLIO & SECURITY (pf_)
# ==========================================

class User(Base):
    __tablename__ = "pf_users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    password_hash = Column(String)
    role = Column(String)
    full_name = Column(String)

class Project(Base):
    __tablename__ = "pf_projects"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    title = Column(String, index=True)
    category = Column(String)
    description = Column(Text) # Ini akan menjadi subtitle
    tech_stack = Column(String)
    # Tambahkan Kolom Baru Berikut:
    problem_statement = Column(Text, nullable=True) # Untuk narasi masalah
    objective = Column(Text, nullable=True)         # Untuk tujuan riset
    approach = Column(Text, nullable=True)          # Untuk metodologi/arsitektur
    metrics_json = Column(Text, nullable=True)      # Menyimpan label & value (Format JSON)
    flowchart_json = Column(Text, nullable=True)
    
    thumbnail_url = Column(String, nullable=True) 
    documentation_url = Column(String, nullable=True)
    has_live_demo = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class AuditLog(Base):
    """Log aktivitas keamanan sistem."""
    __tablename__ = "pf_audit_logs"
    id = Column(Integer, primary_key=True, index=True)
    action = Column(String(100))
    username = Column(String(50))
    details = Column(Text)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

# ==========================================
# 2. RICE SUPPLY CHAIN MODELS (rice_)
# ==========================================

class RiceProduction(Base):
    """Data produksi beras berdasarkan production_data.csv."""
    __tablename__ = "rice_production"
    id = Column(Integer, primary_key=True, index=True)
    provinsi = Column(String(100), index=True)
    tahun = Column(Integer, index=True)
    luas_panen_ha = Column(Float)
    produksi_ton = Column(Float)
    suhu_min = Column(Float)
    suhu_rata = Column(Float)
    suhu_maks = Column(Float)
    kelembapan_persen = Column(Float)
    curah_hujan_mm = Column(Float)
    hari_hujan = Column(Float)
    kecepatan_angin = Column(Float)
    tekanan_udara = Column(Float)
    penyinaran_matahari = Column(Float)
    annual_soi_bom = Column(Float)
    soi_bom_label = Column(String(50))
    annual_dmi = Column(Float)
    produktivitas_ton_ha = Column(Float)
    data_label = Column(String(20)) # 'original' / 'prediction'
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class RiceDemand(Base):
    """Data permintaan beras berdasarkan demand_data.csv."""
    __tablename__ = "rice_demand"
    id = Column(Integer, primary_key=True, index=True)
    provinsi = Column(String(100), index=True)
    tahun = Column(Integer, index=True)
    total_konsumsi_ton = Column(Float)
    jumlah_penduduk = Column(Float)
    laju_pertumbuhan = Column(Float)
    harga_beras = Column(Float)
    harga_jagung = Column(Float)
    harga_mie_instan = Column(Float)
    tingkat_kemiskinan = Column(Float)
    garis_kemiskinan = Column(Float)
    pdrb_perkapita = Column(Float)
    gini_rasio = Column(Float)
    rata_anggota_rt = Column(Float)
    tpt_persen = Column(Float)
    data_label = Column(String(20))
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class RiceDivrePrediction(Base):
    """Hasil prediksi agregat per wilayah DIVRE (divre_predictions.csv)."""
    __tablename__ = "rice_divre_predictions"
    id = Column(Integer, primary_key=True, index=True)
    divre_name = Column(String(100), index=True)
    base_year = Column(Integer)
    horizon_years = Column(Integer)
    target_year = Column(Integer)
    predicted_production = Column(Float)
    predicted_demand = Column(Float)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

# ==========================================
# 3. MANGROVE RESEARCH MODELS (mng_)
# ==========================================

class MangrovePemetaan(Base):
    """Dataset Pemetaan (data_final_untuk_model & klasifikasi_titik_dengan_model)."""
    __tablename__ = "mng_pemetaan"
    id = Column(Integer, primary_key=True, index=True)
    latitude = Column(Float)
    longitude = Column(Float)
    # Spectral Bands
    b1 = Column(Float); b2 = Column(Float); b3 = Column(Float); b4 = Column(Float)
    b5 = Column(Float); b6 = Column(Float); b7 = Column(Float); b8 = Column(Float)
    b8a = Column(Float); b9 = Column(Float); b11 = Column(Float); b12 = Column(Float)
    # Indices
    ndvi = Column(Float); evi = Column(Float); savi = Column(Float)
    # Labels
    ground_truth_class = Column(String(50)) # 'Class' asli
    predicted_class = Column(String(50))    # Hasil prediksi model pemetaan

class MangrovePrediksi(Base):
    """Dataset Prediksi (klasifikasi_dengan_spesies_dbh & output_prediksi_agb_agc)."""
    __tablename__ = "mng_prediksi"
    id = Column(Integer, primary_key=True, index=True)
    latitude = Column(Float); longitude = Column(Float)
    x_coord = Column(Float); y_coord = Column(Float)
    acquisition_date = Column(String(100))
    # Spectral Metadata
    ndvi = Column(Float); evi = Column(Float); savi = Column(Float)
    # Species & DBH
    predicted_class = Column(String(50))
    dominant_species = Column(String(100))
    species_code = Column(Integer)
    actual_dbh = Column(Float)
    predicted_dbh_cm = Column(Float)
    # Carbon Estimation Results
    agb_per_tree_kg = Column(Float)
    agc_per_tree_kg = Column(Float)
    predicted_agb_kg = Column(Float)
    predicted_agc_kg = Column(Float)
    log_dbh = Column(Float)
    
class ResumeData(Base):
    """Penyimpanan data CV dinamis untuk PDF Generator."""
    __tablename__ = "pf_resume_data"
    id = Column(Integer, primary_key=True, index=True)
    resume_name = Column(String(100), default="My Resume") # Tambahkan ini
    is_active = Column(Boolean, default=False)             # Tambahkan ini
    full_name = Column(String(100))
    email = Column(String(100))
    phone = Column(Text)          
    linkedin_url = Column(Text)   
    github_url = Column(Text)
    summary = Column(Text)
    experience_json = Column(Text)  
    projects_json = Column(Text)
    education_json = Column(Text)
    skills_json = Column(Text)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    
class Rating(Base):
    __tablename__ = "pf_ratings"
    id = Column(Integer, primary_key=True, index=True)
    score = Column(Integer)
    comment = Column(Text, nullable=True)
    category = Column(String(50))
    user_ip = Column(String(45)) # Menambahkan kolom untuk menyimpan IP (IPv4 atau IPv6)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)