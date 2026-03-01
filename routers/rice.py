import sys
import os
import pandas as pd
import numpy as np
import math
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict
from database import SessionLocal
import models

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from ml_models.penelitian.pipeline.pipeline_demand import run_demand
    from ml_models.penelitian.pipeline.pipeline_production import run_production
except ImportError as e:
    print(f"WARNING: Pipeline AI tidak dapat dimuat. Error: {e}")

router = APIRouter(prefix="/api/v1/rice", tags=["Rice Supply Chain Analytics"])

PRODUCTION_ADJUSTMENT_FACTOR = 0.6402

def get_db():
    db = SessionLocal()
    try: yield db
    finally: db.close()

# --- 1. SCHEMAS LENGKAP DENGAN ALIAS ---
class PredictProductionInput(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    tahun: int
    provinsi: str
    luas_panen: Optional[float] = Field(None, alias="luas panen (ha)")
    suhu_min: Optional[float] = Field(None, alias="suhu min (oc)")
    suhu_rata: Optional[float] = Field(None, alias="suhu rata-rata (oc)")
    suhu_maks: Optional[float] = Field(None, alias="suhu maks (oc)")
    kelembapan: Optional[float] = Field(None, alias="kelembapan (%)")
    curah_hujan: Optional[float] = Field(None, alias="curah hujan (mm)")
    hari_hujan: Optional[float] = Field(None, alias="jumlah hari hujan (hari)")
    kecepatan_angin: Optional[float] = Field(None, alias="kecepatan angin (m/s)")
    tekanan_udara: Optional[float] = Field(None, alias="tekanan udara (mb)")
    penyinaran: Optional[float] = Field(None, alias="penyinaran matahari (%)")
    soi_bom: Optional[float] = Field(None, alias="annual_soi_bom")
    
    # 🔴 PERBAIKAN: Tambahkan soi_label (Sangat Penting untuk OHE)
    soi_label: Optional[str] = Field(None, alias="soi_label")
    
    dmi: Optional[float] = Field(None, alias="annual_dmi")
    produktivitas: Optional[float] = Field(None, alias="produktivitas")

class PredictDemandInput(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    tahun: int
    provinsi: str
    populasi: Optional[float] = Field(None, alias="Jumlah Penduduk (Jiwa)")
    pertumbuhan_pop: Optional[float] = Field(None, alias="Laju Pertumbuhan Penduduk (%)")
    harga_beras: Optional[float] = Field(None, alias="Harga Beras (Rp/Kg)")
    harga_jagung: Optional[float] = Field(None, alias="Harga Jagung (Rp/Kg)")
    harga_mi: Optional[float] = Field(None, alias="Harga Mie Instan (Rp/Bungkus)")
    kemiskinan: Optional[float] = Field(None, alias="Tingkat Kemiskinan (%)")
    garis_kemiskinan: Optional[float] = Field(None, alias="Garis Kemiskinan per Kapita (Rp/tahun)")
    pdrb: Optional[float] = Field(None, alias="Produk Domestik Regional Bruto per Kapita HK (Ribu Rp)")
    gini: Optional[float] = Field(None, alias="Gini Rasio")
    ukuran_rt: Optional[float] = Field(None, alias="Rata-rata Banyaknya Anggota Rumah Tangga (Jiwa)")
    pengangguran: Optional[float] = Field(None, alias="Tingkat Pengangguran Terbuka (%)")
    produksi_dari_model: float = Field(..., alias="Produksi (ton)")

class HistoryData(BaseModel):
    tahun: int
    produksi: Optional[float] = None
    demand: Optional[float] = None

# --- HELPER: SAFE CLEAN NaN ---
def clean_nan(val):
    try:
        if val is None or pd.isna(val) or math.isnan(float(val)) or math.isinf(float(val)):
            return None
        return float(val)
    except:
        return None

# --- HELPER: AMBIL HISTORY 12 TAHUN (Exact Match) ---
def get_historical_data_from_db(db: Session, provinsi: str, target_year: int, lookback: int = 12) -> pd.DataFrame:
    start_year = target_year - lookback
    end_year = target_year - 1
    
    demands = db.query(models.RiceDemand).filter(
        models.RiceDemand.provinsi.ilike(provinsi),
        models.RiceDemand.tahun >= start_year, models.RiceDemand.tahun <= end_year
    ).order_by(models.RiceDemand.tahun.asc()).all()
    
    productions = db.query(models.RiceProduction).filter(
        models.RiceProduction.provinsi.ilike(provinsi),
        models.RiceProduction.tahun >= start_year, models.RiceProduction.tahun <= end_year
    ).order_by(models.RiceProduction.tahun.asc()).all()
    
    prod_map = {p.tahun: p.produksi_ton for p in productions}
    data_list = []
    
    for d in demands:
        row = {
            "Tahun": float(d.tahun), "Provinsi": d.provinsi,
            "Jumlah Penduduk (Jiwa)": float(getattr(d, 'jumlah_penduduk', np.nan) or np.nan),
            "Laju Pertumbuhan Penduduk (%)": float(getattr(d, 'laju_pertumbuhan', np.nan) or np.nan),
            "Harga Beras (Rp/Kg)": float(getattr(d, 'harga_beras', np.nan) or np.nan),
            "Harga Jagung (Rp/Kg)": float(getattr(d, 'harga_jagung', np.nan) or np.nan),
            "Harga Mie Instan (Rp/Bungkus)": float(getattr(d, 'harga_mie_instan', np.nan) or np.nan),
            "Tingkat Kemiskinan (%)": float(getattr(d, 'tingkat_kemiskinan', np.nan) or np.nan),
            "Garis Kemiskinan per Kapita (Rp/tahun)": float(getattr(d, 'garis_kemiskinan', np.nan) or np.nan),
            "Produk Domestik Regional Bruto per Kapita HK (Ribu Rp)": float(getattr(d, 'pdrb_perkapita', np.nan) or np.nan),
            "Gini Rasio": float(getattr(d, 'gini_rasio', np.nan) or np.nan),
            "Rata-rata Banyaknya Anggota Rumah Tangga (Jiwa)": float(getattr(d, 'rata_anggota_rt', np.nan) or np.nan),
            "Tingkat Pengangguran Terbuka (%)": float(getattr(d, 'tpt_persen', getattr(d, 'tpt', np.nan)) or np.nan),
            "Produksi (ton)": float(prod_map.get(d.tahun, np.nan))
        }
        data_list.append(row)
    return pd.DataFrame(data_list)

# ==========================================
# ENDPOINTS API
# ==========================================

@router.get("/baseline/{provinsi}")
def get_baseline_input(provinsi: str, db: Session = Depends(get_db)):
    """Mengambil baseline tahun terakhir secara presisi"""
    prod = db.query(models.RiceProduction).filter(models.RiceProduction.provinsi.ilike(provinsi)).order_by(models.RiceProduction.tahun.desc()).first()
    dem = db.query(models.RiceDemand).filter(models.RiceDemand.provinsi.ilike(provinsi)).order_by(models.RiceDemand.tahun.desc()).first()

    return {
        "production": {
            "luas_panen": clean_nan(getattr(prod, "luas_panen_ha", None)),
            "produktivitas": clean_nan(getattr(prod, "produktivitas_ton_ha", None)),
            "suhu_min": clean_nan(getattr(prod, "suhu_min", None)),
            "suhu_rata": clean_nan(getattr(prod, "suhu_rata", None)),
            "suhu_maks": clean_nan(getattr(prod, "suhu_maks", None)),
            "kelembapan": clean_nan(getattr(prod, "kelembapan_persen", None)),
            "curah_hujan": clean_nan(getattr(prod, "curah_hujan_mm", None)),
            "hari_hujan": clean_nan(getattr(prod, "hari_hujan", None)),
            "kecepatan_angin": clean_nan(getattr(prod, "kecepatan_angin", None)),
            "tekanan_udara": clean_nan(getattr(prod, "tekanan_udara", None)),
            "penyinaran": clean_nan(getattr(prod, "penyinaran_matahari", None)),
            "soi_bom": clean_nan(getattr(prod, "annual_soi_bom", getattr(prod, "annual_soi", None))),
            "soi_label": getattr(prod, "soi_bom_label", getattr(prod, "soi_label", "Netral")), # 🔴 TAMBAHAN
            "dmi": clean_nan(getattr(prod, "annual_dmi", None)),
        },
        "demand": {
            "populasi": clean_nan(getattr(dem, "jumlah_penduduk", None)),
            "pertumbuhan_pop": clean_nan(getattr(dem, "laju_pertumbuhan", None)),
            "harga_beras": clean_nan(getattr(dem, "harga_beras", None)),
            "harga_jagung": clean_nan(getattr(dem, "harga_jagung", None)),
            "harga_mi": clean_nan(getattr(dem, "harga_mie_instan", getattr(dem, "harga_mie", None))),
            "kemiskinan": clean_nan(getattr(dem, "tingkat_kemiskinan", None)),
            "garis_kemiskinan": clean_nan(getattr(dem, "garis_kemiskinan", None)),
            "pdrb": clean_nan(getattr(dem, "pdrb_perkapita", None)),
            "gini": clean_nan(getattr(dem, "gini_rasio", None)),
            "ukuran_rt": clean_nan(getattr(dem, "rata_anggota_rt", None)),
            "pengangguran": clean_nan(getattr(dem, "tpt_persen", getattr(dem, "tpt", None)))
        }
    }

@router.get("/history/{provinsi}", response_model=List[HistoryData])
def get_provinsi_history(provinsi: str, db: Session = Depends(get_db)):
    demands = db.query(models.RiceDemand).filter(
        models.RiceDemand.provinsi.ilike(provinsi), models.RiceDemand.tahun >= 2007
    ).order_by(models.RiceDemand.tahun.asc()).all()
    
    productions = db.query(models.RiceProduction).filter(
        models.RiceProduction.provinsi.ilike(provinsi), models.RiceProduction.tahun >= 2007
    ).order_by(models.RiceProduction.tahun.asc()).all()
    
    prod_map = {p.tahun: p.produksi_ton for p in productions}
    history = []
    
    for d in demands:
        raw_prod = prod_map.get(d.tahun, None)
        prod_beras = float(raw_prod) * PRODUCTION_ADJUSTMENT_FACTOR if raw_prod is not None else None
        
        history.append({
            "tahun": d.tahun,
            "produksi": clean_nan(prod_beras),
            "demand": clean_nan(getattr(d, 'total_konsumsi_ton', getattr(d, 'total_konsumsi', None)))
        })
    return history

@router.post("/predict/production")
def simulate_production(input_data: PredictProductionInput, db: Session = Depends(get_db)):
    raw_input = input_data.model_dump(by_alias=True, exclude_none=True)
    raw_input["provinsi"] = input_data.provinsi
    raw_input["tahun"] = input_data.tahun
    raw_input["produksi (ton)"] = 0.0 
    
    try:
        result = run_production(raw_input)
        prod_gabah = float(result.get("prediksi_produksi", 0.0))
        prod_beras = prod_gabah * PRODUCTION_ADJUSTMENT_FACTOR
        
        # --- LOGIKA REPLACE / UPSERT PRODUKSI ---
        existing = db.query(models.RiceDivrePrediction).filter(
            models.RiceDivrePrediction.divre_name.ilike(input_data.provinsi),
            models.RiceDivrePrediction.target_year == input_data.tahun
        ).first()

        if existing:
            # Jika sudah ada, update nilai produksinya saja
            existing.predicted_production = prod_gabah
        else:
            # Jika belum ada, buat baris baru
            new_pred = models.RiceDivrePrediction(
                divre_name=input_data.provinsi,
                base_year=2024,
                target_year=input_data.tahun,
                predicted_production=prod_gabah,
                predicted_demand=0.0 # Default awal sebelum hitung demand
            )
            db.add(new_pred)
        
        db.commit()
        return {
            "production_gabah": clean_nan(prod_gabah),
            "production_beras": clean_nan(prod_beras)
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict/demand")
def simulate_demand(input_data: PredictDemandInput, db: Session = Depends(get_db)):
    raw_input = input_data.model_dump(by_alias=True, exclude_none=True)
    raw_input["Provinsi"] = input_data.provinsi
    raw_input["Tahun"] = input_data.tahun

    try:
        hist_df = get_historical_data_from_db(db, input_data.provinsi, input_data.tahun)
        raw_input['__historical_df_override'] = hist_df
        result = run_demand(raw_input)
        dem_value = float(result.get("prediksi_permintaan", 0.0))
        prod_value = float(input_data.produksi_dari_model) # Ini adalah Gabah

        # --- LOGIKA REPLACE / UPSERT DEMAND ---
        existing = db.query(models.RiceDivrePrediction).filter(
            models.RiceDivrePrediction.divre_name.ilike(input_data.provinsi),
            models.RiceDivrePrediction.target_year == input_data.tahun
        ).first()

        if existing:
            existing.predicted_demand = dem_value
            existing.predicted_production = prod_value # Sinkronkan kembali
        else:
            new_pred = models.RiceDivrePrediction(
                divre_name=input_data.provinsi,
                base_year=2024,
                target_year=input_data.tahun,
                predicted_production=prod_value,
                predicted_demand=dem_value
            )
            db.add(new_pred)
        
        db.commit()
        return {"demand": clean_nan(dem_value), "status": "REPLACED_IN_DB"}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get("/chart-data/{provinsi}")
def get_combined_chart_data(provinsi: str, db: Session = Depends(get_db)):
    # 1. Batasi data historis hanya sampai tahun 2024 (Baseline terakhir yang valid)
    history_raw = db.query(models.RiceDemand).filter(
        models.RiceDemand.provinsi.ilike(provinsi), 
        models.RiceDemand.tahun >= 2007,
        models.RiceDemand.tahun <= 2024 # Batasi di sini
    ).order_by(models.RiceDemand.tahun.asc()).all()
    
    productions_raw = db.query(models.RiceProduction).filter(
        models.RiceProduction.provinsi.ilike(provinsi), 
        models.RiceProduction.tahun >= 2007,
        models.RiceProduction.tahun <= 2024 # Batasi di sini
    ).all()
    
    prod_map = {p.tahun: p.produksi_ton for p in productions_raw}
    combined_data = []

    # 2. Masukkan data historis (Garis Solid)
    for d in history_raw:
        raw_prod = prod_map.get(d.tahun, None)
        prod_beras = float(raw_prod) * PRODUCTION_ADJUSTMENT_FACTOR if raw_prod is not None else None
        combined_data.append({
            "tahun": d.tahun,
            "produksi": clean_nan(prod_beras),
            "demand": clean_nan(getattr(d, 'total_konsumsi_ton', None)),
            "type": "history"
        })

    # 3. Ambil data prediksi mulai tahun 2025 (Garis Putus-putus)
    predictions = db.query(models.RiceDivrePrediction).filter(
        models.RiceDivrePrediction.divre_name.ilike(provinsi),
        models.RiceDivrePrediction.target_year >= 2025 # Mulai dari 2025
    ).order_by(models.RiceDivrePrediction.target_year.asc()).all()

    for p in predictions:
        val_beras = p.predicted_production * PRODUCTION_ADJUSTMENT_FACTOR
        combined_data.append({
            "tahun": p.target_year,
            "produksi": clean_nan(val_beras), 
            "demand": clean_nan(p.predicted_demand),
            "type": "prediction"
        })
        
    return combined_data