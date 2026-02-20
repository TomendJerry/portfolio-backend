from pathlib import Path
import os
import json
from typing import Dict, Any
BASE_ML_DIR = os.path.dirname(os.path.dirname(__file__))

CONFIG_DIR = os.path.join(BASE_ML_DIR, "config")
SHARED_DIR = os.path.join(BASE_ML_DIR, "shared")

import joblib
import numpy as np
import pandas as pd



def _resolve_ml_path(path: str) -> str:
    """
    Menyelesaikan path relatif aset ML.

    - Kalau path sudah absolut ATAU file-nya sudah ada -> pakai apa adanya
    - Kalau belum ada, coba prefiks "ml/" -> ml/<path>
    """
    if os.path.isabs(path) or os.path.exists(path):
        return path

    alt = os.path.join("ml", path)
    if os.path.exists(alt):
        return alt

    # Biarkan path asli, supaya error berikutnya jelas kalau memang tidak ada
    return path

def resolve_resource_path(relative_path: str) -> str:
    """
    - Kalau sudah ABSOLUT → kembalikan apa adanya
    - Kalau RELATIF      → gabungkan dengan folder ml/
    """
    # Kalau sudah absolut, jangan diapa-apakan
    if os.path.isabs(relative_path):
        return relative_path

    # Base = folder ml (bukan pipeline)
    base = Path(__file__).resolve().parents[1]   # .../ml
    abs_path = base / relative_path
    return str(abs_path)


def load_json(path: str) -> Dict[str, Any]:
    path = resolve_resource_path(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)



def load_pickle(path: str):
    path = resolve_resource_path(path)
    return joblib.load(path)



def load_province_mapping(path: str | None = None):
    """
    Load province mapping dari ml/shared/province_mapping.json
    """
    if path is None or not os.path.isabs(path):
        path = os.path.join(SHARED_DIR, "province_mapping.json")
    return load_json(path)



def resolve_province_key(raw_name: str, mapping: Dict[str, Any]) -> str:
    if raw_name in mapping:
        return raw_name

    lowered = raw_name.strip().lower()
    
    # 1. Exact Match & Alias Match (Pencocokan Persis)
    for key, info in mapping.items():
        if info.get("ui_label", "").strip().lower() == lowered:
            return key
        if info.get("dataset_label", "").strip().lower() == lowered:
            return key
        for alias in info.get("aliases", []):
            if alias.strip().lower() == lowered:
                return key
                
    # 2. PERBAIKAN: Partial Match / Fallback Cerdas
    # Jika UI mengirim "Riau", akan otomatis cocok dengan "Riau & Kepulauan Riau"
    for key in mapping.keys():
        if lowered in key.lower() or key.lower() in lowered:
            print(f"[INFO] Partial match digunakan: '{raw_name}' -> '{key}'")
            return key

    raise ValueError(f"Provinsi '{raw_name}' tidak ditemukan di province_mapping.json")


def get_imputation_folder(province_key: str, mapping: Dict[str, Any]) -> str:
    info = mapping[province_key]
    return info.get("imputation_folder", province_key)


def apply_gain_imputation(df: pd.DataFrame, base_path: str, province_folder: str) -> pd.DataFrame:
    """
    Imputasi missing value per provinsi.

    - Menggunakan col_means.npy + feature_names.json (hasil training GAIN)
    - Nama kolom di df biasanya sudah lowercase, jadi kita cocokan secara
      case-insensitive dengan nama fitur di feature_names.json.
    """
    # Pastikan "" dan " " dianggap NaN
    df_imputed = df.copy()
    df_imputed = df_imputed.replace(["", " "], np.nan)

    # ❗ PERBAIKAN PENTING:
    # base_path dari config (mis. "production_model/imputation")
    # harus dijadikan path ABSOLUT relatif terhadap folder ml/
    base_abs = resolve_resource_path(base_path)
    folder = os.path.join(base_abs, province_folder)

    col_means_path = os.path.join(folder, "col_means.npy")
    feature_names_path = os.path.join(folder, "feature_names.json")

    means_dict = None

    if os.path.exists(col_means_path):
        try:
            loaded = np.load(col_means_path, allow_pickle=True)

            # Case 1: isinya dict
            if loaded.ndim == 0 and isinstance(loaded.item(), dict):
                means_dict = loaded.item()

            # Case 2: array 1D + feature_names.json
            elif loaded.ndim == 1 and os.path.exists(feature_names_path):
                with open(feature_names_path, "r", encoding="utf-8") as f:
                    feature_names = json.load(f)
                if isinstance(feature_names, dict):
                    feature_names = feature_names.get("features", feature_names)

                if len(feature_names) == len(loaded):
                    means_dict = dict(zip(feature_names, loaded))
        except Exception:
            means_dict = None

    # ➜ Terapkan mean per kolom dengan pencocokan lowercase
    if means_dict is not None:
        means_lower = {
            str(k).strip().lower(): float(v)
            for k, v in means_dict.items()
        }

        for col in df_imputed.columns:
            key_l = col.strip().lower()
            if key_l in means_lower:
                df_imputed[col] = df_imputed[col].fillna(means_lower[key_l])

    # Fallback ringan: kalau masih ada NaN, pakai mean dari df sendiri
    if df_imputed.isna().any().any():
        df_imputed = df_imputed.fillna(df_imputed.mean(numeric_only=True))

    return df_imputed

def ensure_columns_order(df: pd.DataFrame, feature_order_path: str) -> pd.DataFrame:
    """
    Menyusun ulang kolom df sesuai urutan fitur yang dipakai saat training model.

    - Kalau ada kolom yang diharapkan model tapi belum ada di df,
      kolom tersebut akan DIBUAT dan diisi 0.0 (bukan raise error).
    """
    feature_order = load_json(feature_order_path)
    if isinstance(feature_order, dict):
        cols = feature_order.get("features", [])
    else:
        cols = feature_order

    # Tambah kolom yang hilang dengan nilai 0.0
    for c in cols:
        if c not in df.columns:
            df[c] = 0.0

    # Opsional: buang kolom yang tidak dipakai model
    df = df[cols]

    return df

def load_feature_order(path: str):
    """
    Membaca urutan fitur dari file JSON.
    Bisa berupa:
    - list langsung: ["feat1", "feat2", ...]
    - dict dengan key "features": {"features": ["feat1", "feat2", ...]}
    """
    data = load_json(path)
    if isinstance(data, dict):
        return data.get("features", [])
    return data


def load_historical_demand_data(path: str) -> pd.DataFrame:
    """
    Membaca data historis permintaan dari Excel.
    Dipakai untuk membangun sequence 12×N fitur.
    """
    return pd.read_excel(resolve_resource_path(path))

    

def load_imputation_components(base_path: str, province_folder: str) -> Dict[str, Any]:
    """
    Loader komponen imputation GAIN per provinsi.

    Struktur folder (contoh):
    base_path/
      └── Jawa Barat/
          ├── feature_names.json
          ├── col_means.npy
          ├── scaler_gain.pkl
          └── gain_generator.h5
    """
    folder = resolve_resource_path(os.path.join(base_path, province_folder))

    feature_names_path = os.path.join(folder, "feature_names.json")
    col_means_path = os.path.join(folder, "col_means.npy")
    scaler_path = os.path.join(folder, "scaler_gain.pkl")
    generator_path = os.path.join(folder, "gain_generator.h5")

    feature_names = load_json(feature_names_path) if os.path.exists(feature_names_path) else None
    col_means = None
    if os.path.exists(col_means_path):
        try:
            col_means = np.load(col_means_path, allow_pickle=True).item()
        except Exception:
            col_means = None

    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

    # Import local supaya tidak bikin error kalau TensorFlow belum siap
    try:
        from tensorflow.keras.models import load_model as _load_model
        generator = _load_model(generator_path) if os.path.exists(generator_path) else None
    except Exception:
        generator = None

    return {
        "folder": folder,
        "feature_names": feature_names,
        "col_means": col_means,
        "scaler": scaler,
        "generator": generator,
    }


def load_production_config(path: str | None = None) -> Dict[str, Any]:
    """
    Load konfigurasi model produksi dari ml/config/production_config.json
    """
    if path is None:
        path = os.path.join(CONFIG_DIR, "production_config.json")
    return load_json(path)


def load_demand_config(path: str | None = None) -> Dict[str, Any]:
    """
    Load konfigurasi model permintaan dari ml/config/demand_config.json
    """
    if path is None:
        path = os.path.join(CONFIG_DIR, "demand_config.json")
    return load_json(path)