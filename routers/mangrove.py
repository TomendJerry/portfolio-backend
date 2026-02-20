import os
import joblib
import pandas as pd
from pathlib import Path  # Gunakan pathlib untuk fleksibilitas OS
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

router = APIRouter(
    prefix="/api/v1/mangrove",
    tags=["Mangrove Mapping & Carbon Research"]
)

# ==========================================
# 1. KONFIGURASI PATH DINAMIS (SOLUSI DEPLOY)
# ==========================================

# Mendapatkan folder tempat file mangrove.py berada (Backend/routers)
CURRENT_DIR = Path(__file__).parent

# Menuju folder ml_models (Naik satu level ke Backend, lalu masuk ke ml_models/skripsi)
BASE_PATH = CURRENT_DIR.parent / "ml_models" / "skripsi"

# Path Modul Pemetaan (Klasifikasi)
PEMETAAN_PATH = BASE_PATH / "pemetaan"
model_mapping = joblib.load(PEMETAAN_PATH / "mangrove_model.pkl")
mapping_features = joblib.load(PEMETAAN_PATH / "model_features.pkl")

# Path Modul Prediksi (Regresi Karbon)
PREDIKSI_PATH = BASE_PATH / "prediksi"
model_agb = joblib.load(PREDIKSI_PATH / "mangrove_agb_model.pkl")
model_agc = joblib.load(PREDIKSI_PATH / "mangrove_agc_model.pkl")
model_dbh = joblib.load(PREDIKSI_PATH / "mangrove_dbh_model.pkl")
regresi_features = joblib.load(PREDIKSI_PATH / "regresi_features.pkl")

# ==========================================
# 2. SCHEMA DATA (PYDANTIC)
# ==========================================

class MappingInput(BaseModel):
    B1: float; B2: float; B3: float; B4: float; B5: float; B6: float
    B7: float; B8: float; B8A: float; B9: float; B11: float; B12: float

class CarbonInput(BaseModel):
    ndvi: float
    evi: float
    species_code: int

# ==========================================
# 3. ENDPOINT API
# ==========================================

@router.post("/predict/mapping")
async def predict_mapping(data: MappingInput):
    try:
        # 1. Konversi ke dictionary dan salin agar data asli aman
        d = data.dict().copy()

        # 2. Normalisasi hanya pada keys yang diawali huruf 'B' (Band)
        for key in d:
            if key.startswith('B'):
                d[key] = d[key] / 10000.0

        # 3. Kalkulasi Indeks (Kunci utama agar tidak error "feature mismatch")
        # NDVI
        ndvi = (d['B8'] - d['B4']) / (d['B8'] + d['B4']) if (d['B8'] + d['B4']) != 0 else 0
        # EVI
        evi_denom = (d['B8'] + 6 * d['B4'] - 7.5 * d['B2'] + 1)
        evi = 2.5 * ((d['B8'] - d['B4']) / evi_denom) if evi_denom != 0 else 0
        # SAVI
        savi = ((d['B8'] - d['B4']) / (d['B8'] + d['B4'] + 0.5)) * 1.5 if (d['B8'] + d['B4'] + 0.5) != 0 else 0

        # 4. Tambahkan indeks hasil hitung ke data
        d['NDVI'] = ndvi
        d['EVI'] = evi
        d['SAVI'] = savi

        # 5. Siapkan DataFrame sesuai urutan fitur model (EVI, NDVI, SAVI)
        try:
            input_df = pd.DataFrame([{
                f: d[f] for f in mapping_features
            }])
        except KeyError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Missing feature: {str(e)}"
            )

        # 6. Prediksi menggunakan Random Forest Classifier
        prediction = model_mapping.predict(input_df)

        return {
            "status": "success",
            "class": str(prediction[0]),
            "indices": {
                "NDVI": round(ndvi, 4),
                "EVI": round(evi, 4),
                "SAVI": round(savi, 4)
            }
        }
    except Exception as e:
        print(f"Error pada Mangrove Mapping: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@router.post("/predict/carbon")
async def predict_carbon(data: CarbonInput):
    """
    Endpoint untuk estimasi Biomassa (AGB), Karbon (AGC), dan Diameter (DBH).
    """
    try:
        # 1. Siapkan DataFrame sesuai urutan fitur [NDVI, EVI, Species_Code]
        input_df = pd.DataFrame([[data.ndvi, data.evi, data.species_code]], 
                                columns=regresi_features)
        
        # 2. Prediksi serentak menggunakan model regresi
        agb = model_agb.predict(input_df)[0]
        agc = model_agc.predict(input_df)[0]
        dbh = model_dbh.predict(input_df)[0]
        
        return {
            "status": "success",
            "results": {
                "agb_kg": round(agb, 2),
                "agc_kg": round(agc, 2),
                "dbh_cm": round(dbh, 2)
            },
            "unit": "kg/tree"
        }
    except Exception as e:
        print(f"Error pada Carbon Prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))